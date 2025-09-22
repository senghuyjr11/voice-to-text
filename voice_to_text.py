import argparse
import json
import os
import queue
import sys
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeout
from datetime import datetime
from typing import Optional

import numpy as np
import sounddevice as sd
from dotenv import load_dotenv
from faster_whisper import WhisperModel

# --------------------------
# Environment & Config
# --------------------------
load_dotenv()

SAMPLE_RATE = int(os.getenv("SAMPLE_RATE", "16000"))
CHANNELS = 1

# Audio device selection
INPUT_DEVICE_INDEX_ENV = os.getenv("INPUT_DEVICE_INDEX")
PREFERRED_DEVICE_KEYWORDS = [
    "headset", "headphone", "earphone", "airpods", "buds", "in-ear",
    "logitech", "hyperx", "steelseries", "razer", "corsair", "sony",
    "samsung", "jabra", "bose", "sennheiser",
]

# Gain and thresholds
PRE_GAIN_DB = float(os.getenv("PRE_GAIN_DB", "10"))
PRE_GAIN = 10 ** (PRE_GAIN_DB / 20.0)

# Whisper
DEVICE = os.getenv("DEVICE", "cpu")        # "cpu" or "cuda"
MODEL_SIZE = os.getenv("MODEL_SIZE", "small")
COMPUTE_TYPE = "float16" if DEVICE == "cuda" else "int8"
ROLLING_SECONDS = float(os.getenv("ROLLING_SECONDS", "8"))
TRANSCRIBE_INTERVAL = float(os.getenv("TRANSCRIBE_INTERVAL", "1.0"))
NO_NEW_TEXT_SECS = float(os.getenv("NO_NEW_TEXT_SECS", "1.8"))

# Gemini
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.0-flash-exp")
GEMINI_TIMEOUT_SECS = float(os.getenv("GEMINI_TIMEOUT_SECS", "12.0"))
GEMINI_RETRIES = int(os.getenv("GEMINI_RETRIES", "2"))

# Output
OUTPUT_DIR = os.getenv("OUTPUT_DIR", "answers")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --------------------------
# Audio Device Utilities
# --------------------------

def list_input_devices():
    devices = sd.query_devices()
    rows = []
    for i, d in enumerate(devices):
        if d.get("max_input_channels", 0) > 0:
            rows.append((i, d["name"], d["max_input_channels"]))
    return rows


def print_available_inputs():
    print("\nAvailable input devices:")
    for idx, name, ch in list_input_devices():
        print(f"  [{idx}] {name}  (in:{ch})")
    print()


def pick_headset_device() -> Optional[int]:
    if INPUT_DEVICE_INDEX_ENV is not None:
        try:
            idx = int(INPUT_DEVICE_INDEX_ENV)
            devs = sd.query_devices()
            if 0 <= idx < len(devs) and devs[idx].get("max_input_channels", 0) > 0:
                print(f"Using INPUT_DEVICE_INDEX from env: [{idx}] {devs[idx]['name']}")
                return idx
        except Exception:
            pass

    candidates = list_input_devices()
    for idx, name, ch in candidates:
        lowered = name.lower()
        if any(k in lowered for k in PREFERRED_DEVICE_KEYWORDS):
            print(f"Auto-selected headset mic: [{idx}] {name}")
            return idx

    try:
        default_in, _ = sd.default.device
        if default_in is not None:
            d = sd.query_devices()[default_in]
            if d.get("max_input_channels", 0) > 0:
                print(f"Using system default input: [{default_in}] {d['name']}")
                return default_in
    except Exception:
        pass

    if candidates:
        idx, name, _ = candidates[0]
        print(f"Fallback to first input device: [{idx}] {name}")
        return idx

    return None


def vu_meter(device_index: int, samplerate: int, seconds: float = 3.0):
    print(f"Running VU meter on device [{device_index}] for {seconds:.1f}s ...")
    clip_warned = False
    with sd.InputStream(
        samplerate=samplerate,
        channels=1,
        device=device_index,
        blocksize=int(samplerate * 0.05),
        dtype="float32",
    ) as stream:
        start = time.time()
        while time.time() - start < seconds:
            audio, _ = stream.read(int(samplerate * 0.05))
            audio = audio * PRE_GAIN
            rms = float(np.sqrt(np.mean(np.square(audio))))
            peak = float(np.max(np.abs(audio)))
            rms_16 = rms * 32768
            peak_16 = peak * 32768
            print(f"\rRMS:{rms_16:6.1f}  Peak:{peak_16:6.1f}", end="", flush=True)
            if peak_16 > 30000 and not clip_warned:
                print("\nWarning: input is near clipping. Reduce mic gain or PRE_GAIN_DB.")
                clip_warned = True
        print()
    print("Aim for RMS > ~80 while speaking. Increase PRE_GAIN_DB or system mic level if needed.\n")


# --------------------------
# Whisper Model
# --------------------------

def load_whisper():
    try:
        print(f"Loading faster-whisper '{MODEL_SIZE}' on {DEVICE} ({COMPUTE_TYPE}) ...")
        return WhisperModel(MODEL_SIZE, device=DEVICE, compute_type=COMPUTE_TYPE)
    except Exception as e:
        print(f"Whisper load warning: {e}. Falling back to CPU int8.")
        return WhisperModel(MODEL_SIZE, device="cpu", compute_type="int8")


model = load_whisper()

# --------------------------
# Agent: Nutrition Extract + Enrichment (Gemini only, no local DB)
# --------------------------

class AgentNutrition:
    """
    Dynamic, agent-driven extractor/enricher.

    Design:
      - No static schemas or examples in code.
      - The agent decides what fields to include.
      - We only rely on a minimal contract for downstream math:
          * foods: array of objects
          * each food should try to include:
                - name (string)
                - per_100g.carbs_g (number)  <-- required for net carbs math
                - optionally: per_100g.{fiber_g,sugars_g,protein_g,fat_g,calories}
                - optionally: per_portion.grams (number)
                - optionally: gi.value (number), gi.confidence (string), gi.source (string)
        Any other fields the agent emits are allowed and simply ignored.
      - Prompts are pulled from env so you can change them without code edits:
          GEMINI_EXTRACTION_PROMPT
          GEMINI_ENRICH_PROMPT
    """

    def __init__(self, api_key: Optional[str], model_name: str):
        self.model = None
        self.genai = None
        if api_key:
            try:
                import google.generativeai as genai
                genai.configure(api_key=api_key)
                self.genai = genai
                # Tip: you can also pass generation_config with response_mime_type="application/json"
                # to reduce the need for manual JSON scraping.
                self.model = genai.GenerativeModel(model_name)
            except Exception as e:
                print(f"Gemini init warning: {e}")
                self.model = None

        self._pool = ThreadPoolExecutor(max_workers=2)

        # Load prompt templates from ENV (no static text baked in).
        # Provide minimal safe defaults if unset.
        self.prompt_extract = os.getenv(
            "GEMINI_EXTRACTION_PROMPT",
            (
                "Goal: From the user's sentence, extract foods and any amounts/size/modifiers mentioned.\n"
                "Return ONLY a single valid JSON object. Do not add prose.\n"
                "The agent may choose any fields that are helpful, but include:\n"
                '- "foods": [ { "name": string, ... } ] at minimum.\n'
                "If possible, infer a broad 'intent' string (free-form).\n"
            )
        )

        self.prompt_enrich = os.getenv(
            "GEMINI_ENRICH_PROMPT",
            (
                "Goal: For the given food name and hints, return ONLY a single valid JSON object with nutrients.\n"
                "Agent decides fields freely, but to enable calculations PLEASE include:\n"
                '- per_100g: { carbs_g: number, optional: fiber_g, sugars_g, protein_g, fat_g, calories }\n'
                "- portion_grams (number) OR per_portion.grams (number) if you can estimate a realistic portion size.\n"
                "- If GI is known, include gi: { value: number, confidence?: string, source?: string }.\n"
                "No prose. Only JSON."
            )
        )

    # ---- core public API
    def ask(self, question: str) -> Optional[dict]:
        base = self._extract_only(question)
        if not base or not isinstance(base.get("foods"), list) or len(base["foods"]) == 0:
            return base  # could be None

        enriched_foods = []
        for f in base["foods"]:
            enriched = self._ensure_enriched_food(f)
            if enriched:
                enriched_foods.append(enriched)
        base["foods"] = enriched_foods
        return base

    # ---- step 1: extraction (agent-driven, no static schema/examples)
    def _extract_only(self, question: str) -> Optional[dict]:
        if not self.model:
            return None

        # Keep it simple: the env-supplied prompt + user text, ask for JSON only.
        prompt = (
            f"{self.prompt_extract}\n"
            "Guidelines:\n"
            "- If nothing edible is found, return {\"foods\": []}.\n"
            "- Use concise field names. Avoid null; prefer omission.\n"
            "- You MAY include extra fields if helpful; they will be ignored if not used.\n\n"
            f"USER:\n{question}\n\nJSON:"
        )
        return self._call_gemini_json(prompt)

    # ---- step 2: enrichment for a single food (agent-driven)
    def _enrich_food(self, name: str, quantity: Optional[dict], size: Optional[str], cooking_method: Optional[str]) -> \
    Optional[dict]:
        if not self.model:
            return None

        # quantity is already normalized dict from _ensure_enriched_food
        hints = {}
        if quantity and (quantity.get("amount") is not None or quantity.get("unit")):
            hints["quantity"] = {"amount": quantity.get("amount"), "unit": quantity.get("unit")}
        if size:
            hints["size"] = size
        if cooking_method:
            hints["cooking_method"] = cooking_method

        prompt = (
            f"{self.prompt_enrich}\n"
            "Input format:\n"
            "{ \"name\": string, \"hints\": { ...optional fields from extraction... } }\n\n"
            "Input:\n"
            f"{json.dumps({'name': name, 'hints': hints}, ensure_ascii=False)}\n\n"
            "JSON:"
        )
        return self._call_gemini_json(prompt)

    def _ensure_enriched_food(self, food: dict) -> Optional[dict]:
        name = (food.get("name") or "").strip()
        if not name:
            return None

        # NEW: normalize quantity into {'amount', 'unit'}
        raw_quantity = food.get("quantity")
        quantity = normalize_quantity(raw_quantity)

        size = food.get("size")
        method = food.get("cooking_method")

        enriched = self._enrich_food(name, quantity, size, method)
        if not enriched:
            return None

        merged = {
            "name": enriched.get("name", name),
            "brand": food.get("brand"),
            "quantity": quantity if (quantity.get("amount") is not None or quantity.get("unit")) else None,
            "size": size,
            "cooking_method": method,
            "portion_grams": enriched.get("portion_grams"),
            "per_100g": enriched.get("per_100g"),
            "per_portion": enriched.get("per_portion"),
            "gi": enriched.get("gi"),
        }

        # Light normalization if model used another shape
        if not merged.get("portion_grams"):
            portion = enriched.get("portion") or {}
            grams = portion.get("grams") if isinstance(portion, dict) else None
            if grams is not None:
                merged["portion_grams"] = grams

        return merged

    # ---- Gemini helpers (timeout + retries) ----
    def _call_gemini_json(self, prompt: str) -> Optional[dict]:
        if not self.model:
            return None

        last_err = None
        for attempt in range(GEMINI_RETRIES + 1):
            try:
                fut = self._pool.submit(self.model.generate_content, prompt)
                out = fut.result(timeout=GEMINI_TIMEOUT_SECS)
                # Prefer JSON-mode if available; otherwise, robust scrape.
                txt = getattr(out, "text", "") or ""
                data = self._extract_json(txt)
                if data and isinstance(data, dict):
                    return data
            except FuturesTimeout as e:
                last_err = e
            except Exception as e:
                last_err = e
            time.sleep(0.4 * (attempt + 1))
        if last_err:
            print(f"Gemini call failed after retries: {last_err}")
        return None

    # ---- helpers
    @staticmethod
    def _extract_json(text: str) -> Optional[dict]:
        # Robustly scan for top-level JSON object(s)
        best = None
        stack = 0
        start = -1
        for i, ch in enumerate(text):
            if ch == "{":
                if stack == 0:
                    start = i
                stack += 1
            elif ch == "}":
                if stack > 0:
                    stack -= 1
                    if stack == 0 and start != -1:
                        candidate = text[start:i+1]
                        try:
                            best = json.loads(candidate)
                        except Exception:
                            pass
        return best


import re

def normalize_quantity(q) -> dict:
    """
    Normalize agent 'quantity' into a dict: {'amount': float|None, 'unit': str|None}.
    Accepts dict, number, string, list; returns {} if unknown.
    Examples:
      1            -> {'amount': 1.0, 'unit': None}
      "1 bowl"     -> {'amount': 1.0, 'unit': 'bowl'}
      {"amount":2,"unit":"pcs"} -> {'amount': 2.0, 'unit': 'pcs'}
      ["1","cup"]  -> {'amount': 1.0, 'unit': 'cup'}
    """
    if q is None:
        return {}

    if isinstance(q, dict):
        return {
            "amount": safe_num(q.get("amount")),
            "unit": (q.get("unit") or "").strip() or None,
        }

    if isinstance(q, (int, float)):
        return {"amount": float(q), "unit": None}

    if isinstance(q, list):
        # Try common patterns like ["1","cup"] or [1,"cup"] or ["1 cup"]
        if len(q) == 1:
            return normalize_quantity(q[0])
        if len(q) >= 2:
            amt = safe_num(q[0])
            unit = " ".join(str(x) for x in q[1:]).strip() or None
            return {"amount": amt, "unit": unit}
        return {}

    if isinstance(q, str):
        s = q.strip()
        # match "1", "1.5", "1 bowl", "2 pcs", "1 medium"
        m = re.match(r"^\s*(\d+(?:\.\d+)?)\s*([^\d].*)?$", s)
        if m:
            amt = safe_num(m.group(1))
            unit = (m.group(2) or "").strip() or None
            return {"amount": amt, "unit": unit}
        # If it's not a number-first string, treat the whole thing as a unit hint.
        return {"amount": None, "unit": s or None}

    return {}

# --------------------------
# Diabetes Metrics from Agent JSON
# --------------------------

def safe_num(x, default=None):
    try:
        return float(x)
    except Exception:
        return default


def pick_portion_grams(food: dict) -> Optional[float]:
    per_portion = food.get("per_portion")
    if isinstance(per_portion, (int, float)):
        return float(per_portion)
    if isinstance(per_portion, dict):
        g = safe_num(per_portion.get("grams"))
        if g:
            return g
    g2 = safe_num(food.get("portion_grams"))
    if g2:
        return g2
    return None


def scale_from_per100(per100: dict, grams: float) -> dict:
    f = grams / 100.0
    return {
        "grams": round(grams, 1),
        "calories": safe_num(per100.get("calories"), 0.0) * f if per100.get("calories") is not None else None,
        "carbs_g": safe_num(per100.get("carbs_g"), 0.0) * f,
        "fiber_g": safe_num(per100.get("fiber_g"), 0.0) * f if per100.get("fiber_g") is not None else 0.0,
        "sugars_g": safe_num(per100.get("sugars_g"), 0.0) * f if per100.get("sugars_g") is not None else None,
        "protein_g": safe_num(per100.get("protein_g"), 0.0) * f if per100.get("protein_g") is not None else None,
        "fat_g": safe_num(per100.get("fat_g"), 0.0) * f if per100.get("fat_g") is not None else None,
    }


def choose_perportion(food: dict) -> Optional[dict]:
    pp = food.get("per_portion")
    if isinstance(pp, (int, float)):
        # Only grams known; other nutrients unknown
        return {"grams": round(float(pp), 1)}
    if isinstance(pp, dict):
        grams = safe_num(pp.get("grams"))
        if grams:
            return {
                "grams": round(grams, 1),
                "calories": safe_num(pp.get("calories")),
                "carbs_g": safe_num(pp.get("carbs_g")),
                "fiber_g": safe_num(pp.get("fiber_g")),
                "sugars_g": safe_num(pp.get("sugars_g")),
                "protein_g": safe_num(pp.get("protein_g")),
                "fat_g": safe_num(pp.get("fat_g")),
            }
    return None


def compute_item_metrics(food: dict) -> Optional[dict]:
    name = food.get("name") or "unknown"
    per100 = food.get("per_100g") or {}
    perportion = choose_perportion(food)
    grams = pick_portion_grams(food)

    if perportion and perportion.get("carbs_g") is not None:
        grams_used = perportion["grams"]
        carbs = safe_num(perportion.get("carbs_g"), 0.0)
        fiber = safe_num(perportion.get("fiber_g"), 0.0)
        sugars = perportion.get("sugars_g")
        protein = perportion.get("protein_g")
        fat = perportion.get("fat_g")
        calories = perportion.get("calories")
    elif grams is not None and per100.get("carbs_g") is not None:
        scaled = scale_from_per100(per100, grams)
        grams_used = scaled["grams"]
        carbs = safe_num(scaled.get("carbs_g"), 0.0)
        fiber = safe_num(scaled.get("fiber_g"), 0.0)
        sugars = scaled.get("sugars_g")
        protein = scaled.get("protein_g")
        fat = scaled.get("fat_g")
        calories = scaled.get("calories")
    else:
        return None

    net = max(carbs - (fiber or 0.0), 0.0)

    gi_block = food.get("gi") or {}
    gi_value = safe_num(gi_block.get("value"))
    gl = gi_value * net / 100.0 if gi_value is not None else None

    return {
        "label": name,
        "grams": round(grams_used, 1),
        "carbs_g": round(carbs, 1),
        "fiber_g": round(fiber or 0.0, 1),
        "sugars_g": round(sugars, 1) if sugars is not None else None,
        "protein_g": round(protein, 1) if protein is not None else None,
        "fat_g": round(fat, 1) if fat is not None else None,
        "calories": round(calories, 0) if calories is not None else None,
        "net_carbs_g": round(net, 1),
        "gi": gi_value,
        "gi_confidence": gi_block.get("confidence"),
        "gl": round(gl, 1) if gl is not None else None,
        "source": gi_block.get("source"),
    }


def evaluate_from_agent(agent_json: dict):
    foods = agent_json.get("foods") or []
    items = []
    for f in foods:
        m = compute_item_metrics(f)
        if m:
            items.append(m)

    if not items:
        return [], {"total_net_carbs_g": 0.0, "meal_gl": None}, "No computable nutrients returned by the agent."

    total_net = round(sum(i["net_carbs_g"] for i in items), 1)
    gl_vals = [i["gl"] for i in items if i["gl"] is not None]
    meal_gl = round(sum(gl_vals), 1) if gl_vals else None

    if meal_gl is None:
        verdict = "GI data missing for at least one item. Showing net carbs."
    elif meal_gl <= 10:
        verdict = "Low glycemic load (≤10)."
    elif meal_gl <= 19:
        verdict = "Moderate glycemic load (11–19)."
    elif meal_gl <= 29:
        verdict = "Moderately high glycemic load (20–29) — consider smaller carb portion or more protein/veg."
    else:
        verdict = "High glycemic load (≥30) — reduce carbs or balance with protein/veg and light activity."

    return items, {"total_net_carbs_g": total_net, "meal_gl": meal_gl}, verdict


# --------------------------
# Pretty console formatter (one question → one answer)
# --------------------------

def _fmt_value(v, unit=""):
    if v is None:
        return "—"
    try:
        if isinstance(v, float) and v.is_integer():
            v = int(v)
        return f"{v}{unit}"
    except Exception:
        return f"{v}{unit}"


def format_console_answer(question: str, items: list[dict], totals: dict, verdict: str) -> str:
    lines: list[str] = []
    lines.append(f"Processing: {question}")
    lines.append("")
    # Use the first item as the headline if present
    focus = items[0] if items else None
    title_name = focus.get("label", "Food") if focus else "Meal"
    lines.append(f"**\"Can I Eat It, {title_name}?\" (Nutrition)**")
    lines.append("")

    # Per-item sections (first item emphasized)
    for idx, it in enumerate(items[:3]):
        name = it.get("label", "Food")
        lines.append(f"**Key Nutrition Facts (per ~{_fmt_value(it.get('grams'),' g')} portion of {name}):**")
        lines.append("")
        lines.append(f"*   Calories: {_fmt_value(it.get('calories'),' kcal')}")
        lines.append(f"*   Carbohydrates: {_fmt_value(it.get('carbs_g'),' g')} (sugars: {_fmt_value(it.get('sugars_g'),' g')}, fiber: {_fmt_value(it.get('fiber_g'),' g')})")
        lines.append(f"*   Protein: {_fmt_value(it.get('protein_g'),' g')}")
        lines.append(f"*   Fat: {_fmt_value(it.get('fat_g'),' g')}")
        gi = it.get("gi")
        gi_conf = it.get("gi_confidence")
        if gi is not None:
            conf = f" ({gi_conf} confidence)" if gi_conf else ""
            lines.append(f"*   GI: {_fmt_value(gi)}{conf}; GL for this portion: {_fmt_value(it.get('gl'))}")
        src = it.get("source")
        if src:
            lines.append(f"*   GI source: {src}")
        lines.append("")


    # Verdict & totals
    tnet = totals.get("total_net_carbs_g")
    tgl = totals.get("meal_gl")
    gl_part = f", Meal GL ≈ {tgl}" if tgl is not None else ""
    lines.append(f"**In short:** {verdict} Net carbs (meal): ~{_fmt_value(tnet,' g')}{gl_part}.")
    return "\n".join(lines)


# --------------------------
# Audio Processing
# --------------------------
frame_queue: "queue.Queue[np.ndarray]" = queue.Queue()
rolling_buffer = deque(maxlen=int(ROLLING_SECONDS * SAMPLE_RATE))  # store float32


def audio_callback(indata, frames, time_info, status):
    mono = (indata[:, 0].astype(np.float32) * PRE_GAIN).copy()
    try:
        frame_queue.put_nowait(mono)
    except queue.Full:
        pass


def save_answer_json(payload: dict) -> str:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    fname = f"answer_{ts}.json"
    fpath = os.path.join(OUTPUT_DIR, fname)
    with open(fpath, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    return fpath


# --------------------------
# Core loop
# --------------------------

def listen_and_answer(selected_device: int, agent: AgentNutrition, single_shot: bool = True):
    print("Ask your nutrition question... (Ctrl+C to exit)")

    with sd.InputStream(
        samplerate=SAMPLE_RATE,
        channels=CHANNELS,
        device=selected_device,
        blocksize=int(SAMPLE_RATE * 0.03),  # ~30 ms blocks for snappier UI
        dtype="float32",
        callback=audio_callback,
    ):
        while True:
            start_time = time.time()
            current_text = ""
            last_text_change_time = start_time
            last_transcribe_time = 0.0
            print("Start speaking now...")

            try:
                while True:
                    while True:
                        try:
                            chunk = frame_queue.get_nowait()
                            rolling_buffer.extend(chunk)
                        except queue.Empty:
                            break

                    now = time.time()

                    if (now - last_transcribe_time) > TRANSCRIBE_INTERVAL and len(rolling_buffer) > SAMPLE_RATE * 1:
                        last_transcribe_time = now
                        audio = np.array(rolling_buffer, dtype=np.float32)
                        try:
                            segments, _ = model.transcribe(
                                audio,
                                language=None,
                                vad_filter=True,
                                beam_size=1,
                            )
                            new_text = " ".join([s.text for s in segments]).strip()
                            if new_text and new_text != current_text:
                                current_text = new_text
                                last_text_change_time = now
                                print(f"\r{current_text}", end="", flush=True)
                        except Exception:
                            pass

                    if current_text and (now - last_text_change_time) > NO_NEW_TEXT_SECS:
                        break
                    time.sleep(0.01)
            except KeyboardInterrupt:
                raise

            if not current_text:
                if single_shot:
                    return
                else:
                    continue

            print(f"\n\nProcessing: {current_text}")
            started_at = datetime.utcnow().isoformat() + "Z"
            agent_json = agent.ask(current_text) or {"intent": "unknown", "foods": [], "notes": "agent failed or returned empty"}
            items, totals, verdict = evaluate_from_agent(agent_json)

            pretty = format_console_answer(current_text, items, totals, verdict)
            print("\n" + pretty + "\n", flush=True)

            payload = {
                "ok": True if items else False,
                "error": None if items else "No computable nutrients from agent.",
                "engine": "faster-whisper",
                "device": DEVICE,
                "model_size": MODEL_SIZE,
                "sample_rate": SAMPLE_RATE,
                "rolling_seconds": ROLLING_SECONDS,
                "pre_gain_db": PRE_GAIN_DB,
                "question": current_text,
                "agent_raw": agent_json,
                "items": items,
                "totals": totals,
                "verdict": verdict,
                "answer_text": pretty,
                "started_at": started_at,
                "finished_at": datetime.utcnow().isoformat() + "Z",
            }
            out_path = save_answer_json(payload)
            print(f"Saved JSON: {out_path}\n", flush=True)

            rolling_buffer.clear()
            if single_shot:
                return


# --------------------------
# Text-mode (for quick testing without audio)
# --------------------------

def run_text_mode(text: str, agent: AgentNutrition):
    print(f"Text mode input: {text}")
    started_at = datetime.utcnow().isoformat() + "Z"
    agent_json = agent.ask(text) or {"intent": "unknown", "foods": [], "notes": "agent failed or returned empty"}
    items, totals, verdict = evaluate_from_agent(agent_json)
    pretty = format_console_answer(text, items, totals, verdict)
    print("\n" + pretty + "\n", flush=True)
    payload = {
        "ok": True if items else False,
        "error": None if items else "No computable nutrients from agent.",
        "engine": "faster-whisper (text-mode)",
        "device": DEVICE,
        "model_size": MODEL_SIZE,
        "sample_rate": SAMPLE_RATE,
        "rolling_seconds": ROLLING_SECONDS,
        "pre_gain_db": PRE_GAIN_DB,
        "question": text,
        "agent_raw": agent_json,
        "items": items,
        "totals": totals,
        "verdict": verdict,
        "answer_text": pretty,
        "started_at": started_at,
        "finished_at": datetime.utcnow().isoformat() + "Z",
    }
    out_path = save_answer_json(payload)
    print(f"Saved JSON: {out_path}\n", flush=True)


# --------------------------
# Main
# --------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Real-time nutrition GL helper")
    parser.add_argument("--text", type=str, default=None, help="Run in text mode with the provided sentence (no audio)")
    parser.add_argument("--continuous", action="store_true", help="Keep listening for multiple questions (default: one-shot)")
    args = parser.parse_args()

    if not GEMINI_API_KEY:
        print("Missing GEMINI_API_KEY. Create a .env file with: GEMINI_API_KEY=your-api-key")
        print("You can still test ASR, but agent-based nutrients require the key.")
        if args.text is None:
            sys.exit(1)

    try:
        agent = AgentNutrition(GEMINI_API_KEY, GEMINI_MODEL)
    except Exception as e:
        print(f"Failed to init agent: {e}")
        sys.exit(1)

    if args.text:
        run_text_mode(args.text, agent)
        sys.exit(0)

    selected = pick_headset_device()
    if selected is None:
        print_available_inputs()
        print("No input device available. Connect your headset mic or set INPUT_DEVICE_INDEX, then retry.")
        sys.exit(1)

    print_available_inputs()
    try:
        vu_meter(selected, SAMPLE_RATE, seconds=3.0)
    except Exception as e:
        print(f"VU meter error (continuing anyway): {e}")

    try:
        listen_and_answer(selected, agent, single_shot=(not args.continuous))
    except KeyboardInterrupt:
        print("\nExiting.")