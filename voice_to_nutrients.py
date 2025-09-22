import argparse
import json
import os
import queue
import re
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

# ==========================
# Environment & Config
# ==========================
load_dotenv()

# Audio basics
SAMPLE_RATE = int(os.getenv("SAMPLE_RATE", "16000"))
CHANNELS = int(os.getenv("CHANNELS", "1"))
PRE_GAIN_DB = float(os.getenv("PRE_GAIN_DB", "10"))
PRE_GAIN = 10 ** (PRE_GAIN_DB / 20.0)

# Stream timing
ROLLING_SECONDS = float(os.getenv("ROLLING_SECONDS", "8"))
TRANSCRIBE_INTERVAL = float(os.getenv("TRANSCRIBE_INTERVAL", "1.0"))
NO_NEW_TEXT_SECS = float(os.getenv("NO_NEW_TEXT_SECS", "1.8"))
STREAM_BLOCK_MS = float(os.getenv("STREAM_BLOCK_MS", "30"))
MIN_SEC_BEFORE_TRANSCRIBE = float(os.getenv("MIN_SEC_BEFORE_TRANSCRIBE", "1.0"))

# Whisper
DEVICE = os.getenv("DEVICE", "cpu")
MODEL_SIZE = os.getenv("MODEL_SIZE", "small")
COMPUTE_TYPE = os.getenv("COMPUTE_TYPE", ("float16" if DEVICE == "cuda" else "int8"))
WHISPER_BEAM_SIZE = int(os.getenv("WHISPER_BEAM_SIZE", "1"))

# Gemini
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL = os.getenv("GEMINI_MODEL")
GEMINI_TIMEOUT_SECS = float(os.getenv("GEMINI_TIMEOUT_SECS", "12.0"))
GEMINI_RETRIES = int(os.getenv("GEMINI_RETRIES", "2"))
GEMINI_BACKOFF_SECS = float(os.getenv("GEMINI_BACKOFF_SECS", "0.4"))

# Prompts (edit in .env, not in code)
PROMPT_EXTRACT = os.getenv(
    "GEMINI_EXTRACTION_PROMPT",
    "From the user's sentence, extract foods mentioned. Return ONLY one JSON object with a top-level array 'foods' of objects (at least 'name'). Avoid null; omit unknown fields."
)
PROMPT_ENRICH = os.getenv(
    "GEMINI_ENRICH_PROMPT",
    "Given a food and optional hints, return ONLY one JSON object with nutrients. Include per_100g (carbs_g required; optionally fiber_g,sugars_g,protein_g,fat_g,calories). If you can, include per_portion with grams and nutrients. If GI known, include gi.value and optionally confidence/source."
)

# Output
OUTPUT_DIR = os.getenv("OUTPUT_DIR", "answers")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Display
MAX_ITEMS_SHOWN = int(os.getenv("MAX_ITEMS_SHOWN", "3"))
ROUND_GRAMS_DECIMALS = int(os.getenv("ROUND_GRAMS_DECIMALS", "1"))
ROUND_NUTRIENT_DECIMALS = int(os.getenv("ROUND_NUTRIENT_DECIMALS", "1"))
ROUND_CALORIES_DECIMALS = int(os.getenv("ROUND_CALORIES_DECIMALS", "0"))

# GI Source (shown even if agent omits it)
GI_SOURCE_DEFAULT = os.getenv("GI_SOURCE_DEFAULT")

# ==========================
# Audio device helpers
# ==========================
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
    env_idx = os.getenv("INPUT_DEVICE_INDEX")
    if env_idx is not None:
        try:
            idx = int(env_idx)
            devs = sd.query_devices()
            if 0 <= idx < len(devs) and devs[idx].get("max_input_channels", 0) > 0:
                print(f"Using INPUT_DEVICE_INDEX from env: [{idx}] {devs[idx]['name']}")
                return idx
        except Exception:
            pass
    # prefer headset-like names
    preferred = [s.strip().lower() for s in os.getenv(
        "PREFERRED_DEVICE_KEYWORDS",
        "headset, headphone, earphone, airpods, buds, in-ear, logitech, hyperx, steelseries, razer, corsair, sony, samsung, jabra, bose, sennheiser"
    ).split(",") if s.strip()]
    candidates = list_input_devices()
    for idx, name, ch in candidates:
        lowered = name.lower()
        if any(k in lowered for k in preferred):
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

# ==========================
# Whisper loader
# ==========================
def load_whisper():
    try:
        print(f"Loading faster-whisper '{MODEL_SIZE}' on {DEVICE} ({COMPUTE_TYPE}) ...")
        return WhisperModel(MODEL_SIZE, device=DEVICE, compute_type=COMPUTE_TYPE)
    except Exception as e:
        print(f"Whisper load warning: {e}. Falling back to CPU int8.")
        return WhisperModel(MODEL_SIZE, device="cpu", compute_type="int8")

model = load_whisper()

# ==========================
# Agent (Gemini): extract + enrich
# ==========================
class AgentNutrition:
    def __init__(self, api_key: Optional[str], model_name: str):
        self.model = None
        self.genai = None
        if api_key:
            try:
                import google.generativeai as genai
                genai.configure(api_key=api_key)
                self.genai = genai
                self.model = genai.GenerativeModel(
                    model_name,
                    generation_config={"response_mime_type": "application/json"}
                )
            except Exception as e:
                print(f"Gemini init warning: {e}")
                self.model = None
        self._pool = ThreadPoolExecutor(max_workers=2)

    def ask(self, question: str) -> Optional[dict]:
        """Two-step: extract foods → enrich each with nutrients (agent decides fields)."""
        base = self._extract_only(question)
        if not base or not isinstance(base.get("foods"), list) or len(base["foods"]) == 0:
            return base
        enriched_foods = []
        for f in base["foods"]:
            enriched = self._ensure_enriched_food(f)
            if enriched:
                enriched_foods.append(enriched)
        base["foods"] = enriched_foods
        return base

    def _extract_only(self, question: str) -> Optional[dict]:
        if not self.model:
            return None
        prompt = f"{PROMPT_EXTRACT}\n\nUSER:\n{question}\n\nJSON:"
        return self._call_gemini_json(prompt)

    def _enrich_food(self, name: str, quantity: Optional[dict], size: Optional[str], cooking_method: Optional[str]) -> Optional[dict]:
        if not self.model:
            return None
        hints = {}
        if quantity and (quantity.get("amount") is not None or quantity.get("unit")):
            hints["quantity"] = {"amount": quantity.get("amount"), "unit": quantity.get("unit")}
        if size:
            hints["size"] = size
        if cooking_method:
            hints["cooking_method"] = cooking_method
        prompt = (
            f"{PROMPT_ENRICH}\n\n"
            "Input:\n"
            f"{json.dumps({'name': name, 'hints': hints}, ensure_ascii=False)}\n\n"
            "JSON:"
        )
        return self._call_gemini_json(prompt)

    def _ensure_enriched_food(self, food: dict) -> Optional[dict]:
        name = (food.get("name") or "").strip()
        if not name:
            return None
        quantity = normalize_quantity(food.get("quantity"))
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
            "per_100g": enriched.get("per_100g"),
            "per_portion": enriched.get("per_portion"),
            "gi": enriched.get("gi"),
        }
        return merged

    def _call_gemini_json(self, prompt: str) -> Optional[dict]:
        if not self.model:
            return None
        last_err = None
        for attempt in range(GEMINI_RETRIES + 1):
            try:
                fut = self._pool.submit(self.model.generate_content, prompt)
                out = fut.result(timeout=GEMINI_TIMEOUT_SECS)
                txt = getattr(out, "text", "") or ""
                data = _extract_json(txt)
                if data and isinstance(data, dict):
                    return data
            except FuturesTimeout as e:
                last_err = e
            except Exception as e:
                last_err = e
            time.sleep(GEMINI_BACKOFF_SECS * (attempt + 1))
        if last_err:
            print(f"Gemini call failed after retries: {last_err}")
        return None

# ==========================
# Small helpers
# ==========================
def safe_num(x, default=None):
    try:
        return float(x)
    except Exception:
        return default

def normalize_quantity(q) -> dict:
    if q is None:
        return {}
    if isinstance(q, dict):
        return {"amount": safe_num(q.get("amount")), "unit": (q.get("unit") or "").strip() or None}
    if isinstance(q, (int, float)):
        return {"amount": float(q), "unit": None}
    if isinstance(q, list):
        if len(q) == 1:
            return normalize_quantity(q[0])
        if len(q) >= 2:
            amt = safe_num(q[0])
            unit = " ".join(str(x) for x in q[1:]).strip() or None
            return {"amount": amt, "unit": unit}
        return {}
    if isinstance(q, str):
        s = q.strip()
        m = re.match(r"^\s*(\d+(?:\.\d+)?)\s*([^\d].*)?$", s)
        if m:
            amt = safe_num(m.group(1))
            unit = (m.group(2) or "").strip() or None
            return {"amount": amt, "unit": unit}
        return {"amount": None, "unit": s or None}
    return {}

def _extract_json(text: str) -> Optional[dict]:
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

# ==========================
# Console formatter (no diabetes logic)
# ==========================
def _fmt_value(v, unit=""):
    if v is None:
        return "—"
    try:
        if isinstance(v, float) and v.is_integer():
            v = int(v)
        return f"{v}{unit}"
    except Exception:
        return f"{v}{unit}"

def _round(v, ndigits):
    if v is None:
        return None
    try:
        return round(v, ndigits)
    except Exception:
        return v

def compute_item_metrics_dynamic(food: dict, gi_percent_base: Optional[float]) -> Optional[dict]:
    """
    Compute per-item metrics. GL uses gi_percent_base from agent config.
    If gi_percent_base is None, GL will be omitted (None).
    """
    name = food.get("name") or "unknown"
    per100 = food.get("per_100g") or {}
    pp = food.get("per_portion") or {}
    grams = None
    if isinstance(pp, dict):
        grams = safe_num(pp.get("grams"))

    # Prefer per_portion nutrients when present; else scale per_100g by grams
    if isinstance(pp, dict) and pp.get("carbs_g") is not None:
        grams_used = _round(safe_num(pp.get("grams")), int(os.getenv("ROUND_GRAMS_DECIMALS", "1")))
        carbs = safe_num(pp.get("carbs_g"), 0.0)
        fiber = safe_num(pp.get("fiber_g"), 0.0)
        sugars = safe_num(pp.get("sugars_g"))
        protein = safe_num(pp.get("protein_g"))
        fat = safe_num(pp.get("fat_g"))
        calories = safe_num(pp.get("calories"))
    elif grams is not None and per100.get("carbs_g") is not None:
        # scale from per_100g; denominator is env-configurable (still not a medical threshold)
        denom = float(os.getenv("PER100_DENOM", "100"))
        f = grams / denom
        grams_used = _round(grams, int(os.getenv("ROUND_GRAMS_DECIMALS", "1")))
        carbs = safe_num(per100.get("carbs_g"), 0.0) * f
        fiber = (safe_num(per100.get("fiber_g"), 0.0) * f) if per100.get("fiber_g") is not None else 0.0
        sugars = (safe_num(per100.get("sugars_g"), 0.0) * f) if per100.get("sugars_g") is not None else None
        protein = (safe_num(per100.get("protein_g"), 0.0) * f) if per100.get("protein_g") is not None else None
        fat = (safe_num(per100.get("fat_g"), 0.0) * f) if per100.get("fat_g") is not None else None
        calories = (safe_num(per100.get("calories"), 0.0) * f) if per100.get("calories") is not None else None
    else:
        return None

    net = max((carbs or 0.0) - (fiber or 0.0), 0.0)

    gi_block = food.get("gi") or {}
    gi_value = safe_num(gi_block.get("value"))
    gl = None
    if gi_value is not None and gi_percent_base:
        try:
            gl = gi_value * net / gi_percent_base
        except Exception:
            gl = None

    return {
        "label": name,
        "grams": _round(grams_used, int(os.getenv("ROUND_GRAMS_DECIMALS", "1"))),
        "carbs_g": _round(carbs, int(os.getenv("ROUND_NUTRIENT_DECIMALS", "1"))),
        "fiber_g": _round(fiber or 0.0, int(os.getenv("ROUND_NUTRIENT_DECIMALS", "1"))),
        "sugars_g": _round(sugars, int(os.getenv("ROUND_NUTRIENT_DECIMALS", "1"))) if sugars is not None else None,
        "protein_g": _round(protein, int(os.getenv("ROUND_NUTRIENT_DECIMALS", "1"))) if protein is not None else None,
        "fat_g": _round(fat, int(os.getenv("ROUND_NUTRIENT_DECIMALS", "1"))) if fat is not None else None,
        "calories": _round(calories, int(os.getenv("ROUND_CALORIES_DECIMALS", "0"))) if calories is not None else None,
        "net_carbs_g": _round(net, int(os.getenv("ROUND_NUTRIENT_DECIMALS", "1"))),
        "gi": gi_value,
        "gi_confidence": gi_block.get("confidence"),
        "gl": _round(gl, int(os.getenv("ROUND_NUTRIENT_DECIMALS", "1"))) if gl is not None else None,
        "source": gi_block.get("source"),
    }

def gl_verdict_from_config(meal_gl: Optional[float], cfg: Optional[dict]) -> Optional[str]:
    """
    Categorize GL using bins/messages from agent config.
    Returns a string verdict or None if config/gl missing.
    """
    if meal_gl is None or not cfg:
        return None
    bins = cfg.get("gl_bins")
    msgs = cfg.get("gl_messages")
    if not (isinstance(bins, list) and len(bins) == 3 and isinstance(msgs, list) and len(msgs) == 4):
        return None

    t0, t1, t2 = bins
    t0p1, t1p1, t2p1 = t0 + 1, t1 + 1, t2 + 1

    def fmt(msg):
        return msg.format(t0=t0, t1=t1, t2=t2, t0p1=t0p1, t1p1=t1p1, t2p1=t2p1)

    if meal_gl <= t0:
        return fmt(msgs[0])
    if meal_gl <= t1:
        return fmt(msgs[1])
    if meal_gl <= t2:
        return fmt(msgs[2])
    return fmt(msgs[3])


def evaluate_from_agent(agent_json: dict, agent) -> tuple[list[dict], dict, Optional[str]]:
    """
    Returns (items, totals, verdict). Verdict is None if config not provided by agent.
    """
    cfg = fetch_diabetes_config(agent)  # may be None (we won't invent numbers)

    gi_percent_base = cfg.get("gi_percent_base") if cfg else None

    foods = agent_json.get("foods") or []
    items = []
    for f in foods:
        m = compute_item_metrics_dynamic(f, gi_percent_base)
        if m:
            items.append(m)

    if not items:
        return [], {"total_net_carbs_g": 0.0, "meal_gl": None}, None

    total_net = _round(sum(i["net_carbs_g"] for i in items), int(os.getenv("ROUND_NUTRIENT_DECIMALS", "1")))
    gl_vals = [i["gl"] for i in items if i["gl"] is not None]
    meal_gl = _round(sum(gl_vals), int(os.getenv("ROUND_NUTRIENT_DECIMALS", "1"))) if gl_vals else None

    verdict = gl_verdict_from_config(meal_gl, cfg)
    return items, {"total_net_carbs_g": total_net, "meal_gl": meal_gl}, verdict

def format_console_answer(
    question: str,
    items: list[dict],
    totals: dict,
    verdict: str | None
) -> str:
    """
    Render a concise console answer.
    - Items are already computed (include net_carbs_g, gl, gi, etc.).
    - verdict may be None if the agent didn't provide GL bins/messages.
    """
    lines: list[str] = []
    lines.append(f"Processing: {question}")
    lines.append("")

    title_name = (items[0].get("label") if items else "Meal") or "Meal"
    lines.append(f"**Nutrition Summary — {title_name}**")
    lines.append("")

    # Per-item sections
    for it in items[:int(os.getenv("MAX_ITEMS_SHOWN", "3"))]:
        name = it.get("label", "Food")
        lines.append(f"**{name}**")
        lines.append(f"*   Portion: {_fmt_value(it.get('grams'),' g')}")
        lines.append(f"*   Calories: {_fmt_value(it.get('calories'),' kcal')}")
        lines.append(f"*   Carbs: {_fmt_value(it.get('carbs_g'),' g')} (fiber: {_fmt_value(it.get('fiber_g'),' g')}, sugars: {_fmt_value(it.get('sugars_g'),' g')})")
        lines.append(f"*   Protein: {_fmt_value(it.get('protein_g'),' g')}")
        lines.append(f"*   Fat: {_fmt_value(it.get('fat_g'),' g')}")

        # Diabetes-related numbers are computed dynamically from agent config (no static thresholds in code)
        lines.append(f"*   Net carbs: {_fmt_value(it.get('net_carbs_g'),' g')}")

        gi_val = it.get("gi")
        gi_conf = it.get("gi_confidence")
        if gi_val is not None:
            conf = f" ({gi_conf})" if gi_conf else ""
            lines.append(f"*   GI: {_fmt_value(gi_val)}{conf}")

        if it.get("gl") is not None:
            lines.append(f"*   GL (this portion): {_fmt_value(it.get('gl'))}")

        src = it.get("source") or os.getenv("GI_SOURCE_DEFAULT")
        lines.append(f"*   GI source: {src}")
        lines.append("")

    # Verdict & totals (verdict may be None if agent didn't provide bins/messages)
    tnet = totals.get("total_net_carbs_g")
    tgl = totals.get("meal_gl")
    gl_part = f", Meal GL ≈ {tgl}" if tgl is not None else ""
    prefix = f"{verdict} " if verdict else ""
    lines.append(f"**In short:** {prefix}Net carbs (meal): {_fmt_value(tnet,' g')}{gl_part}.")
    return "\n".join(lines)


# ==========================
# Audio processing
# ==========================
frame_queue: "queue.Queue[np.ndarray]" = queue.Queue()
rolling_buffer = deque(maxlen=int(ROLLING_SECONDS * SAMPLE_RATE))

def audio_callback(indata, frames, time_info, status):
    mono = (indata[:, 0].astype(np.float32) * PRE_GAIN).copy()
    try:
        frame_queue.put_nowait(mono)
    except queue.Full:
        pass

def save_answer_json(payload: dict) -> str:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    fpath = os.path.join(OUTPUT_DIR, f"answer_{ts}.json")
    with open(fpath, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    return fpath

# ==========================
# Core loop
# ==========================
def listen_and_answer(selected_device: int, agent: AgentNutrition, single_shot: bool = True):
    print("Ask your nutrition question... (Ctrl+C to exit)")

    with sd.InputStream(
        samplerate=SAMPLE_RATE,
        channels=CHANNELS,
        device=selected_device,
        blocksize=int(SAMPLE_RATE * (STREAM_BLOCK_MS / 1000.0)),
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
                    # Drain audio queue into rolling buffer
                    while True:
                        try:
                            chunk = frame_queue.get_nowait()
                            rolling_buffer.extend(chunk)
                        except queue.Empty:
                            break

                    now = time.time()

                    # Periodically transcribe rolling buffer
                    if (now - last_transcribe_time) > TRANSCRIBE_INTERVAL and len(rolling_buffer) > SAMPLE_RATE * MIN_SEC_BEFORE_TRANSCRIBE:
                        last_transcribe_time = now
                        audio = np.array(rolling_buffer, dtype=np.float32)
                        try:
                            segments, _ = model.transcribe(
                                audio,
                                language=None,
                                vad_filter=True,
                                beam_size=WHISPER_BEAM_SIZE,
                            )
                            new_text = " ".join([s.text for s in segments]).strip()
                            if new_text and new_text != current_text:
                                current_text = new_text
                                last_text_change_time = now
                                print(f"\r{current_text}", end="", flush=True)
                        except Exception:
                            # swallow transient ASR errors and keep listening
                            pass

                    # If speech has settled for a bit, process it
                    if current_text and (now - last_text_change_time) > NO_NEW_TEXT_SECS:
                        break

                    time.sleep(float(os.getenv("SCHED_YIELD_SECS", "0.01")))
            except KeyboardInterrupt:
                raise

            # Nothing captured? Exit (one-shot) or continue (continuous)
            if not current_text:
                if single_shot:
                    return
                else:
                    continue

            # ===== Process with agent =====
            print(f"\n\nProcessing: {current_text}")
            started_at = datetime.utcnow().isoformat() + "Z"

            agent_json = agent.ask(current_text) or {
                "intent": "unknown",
                "foods": [],
                "notes": "agent failed or returned empty"
            }

            # Dynamic, agent-driven evaluation (no static thresholds in code)
            items, totals, verdict = evaluate_from_agent(agent_json, agent)

            # Render answer
            pretty = format_console_answer(current_text, items, totals, verdict)
            print("\n" + pretty + "\n", flush=True)

            # Persist payload
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

            # Reset for next utterance
            rolling_buffer.clear()
            if single_shot:
                return


# ==========================
# Text mode
# ==========================
def run_text_mode(text: str, agent: AgentNutrition):
    print(f"Text mode input: {text}")
    started_at = datetime.utcnow().isoformat() + "Z"

    # Ask Gemini agent
    agent_json = agent.ask(text) or {
        "intent": "unknown",
        "foods": [],
        "notes": "agent failed or returned empty"
    }

    # Compute nutrition & verdict (agent-driven, no static thresholds)
    items, totals, verdict = evaluate_from_agent(agent_json, agent)

    # Pretty console output
    pretty = format_console_answer(text, items, totals, verdict)
    print("\n" + pretty + "\n", flush=True)

    # Save JSON payload
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


# ===== Dynamic diabetes config (no static numbers) =====

_DIABETES_CFG_CACHE = None

def _as_safe_json(obj) -> str:
    try:
        return json.dumps(obj, ensure_ascii=False)
    except Exception:
        return "{}"

def fetch_diabetes_config(agent) -> Optional[dict]:
    """
    Ask the agent for GL bins/messages and GI scaling base.
    Caches result in-process. Returns dict or None.
    Expected JSON shape:
      {
        "gl_bins": [t0, t1, t2],
        "gl_messages": ["...", "...", "...", "..."],
        "gi_percent_base": 100
      }
    """
    global _DIABETES_CFG_CACHE
    if _DIABETES_CFG_CACHE is not None:
        return _DIABETES_CFG_CACHE

    if not getattr(agent, "model", None):
        return None

    prompt = (os.getenv("DIABETES_CONFIG_PROMPT") or
              "Return ONLY JSON with gl_bins, gl_messages, gi_percent_base.")

    # Optionally provide context about locale/use:
    payload = {"purpose": "glycemic-load categorization for meal outputs"}
    full_prompt = f"{prompt}\n\nINPUT:\n{_as_safe_json(payload)}\n\nJSON:"

    try:
        data = agent._call_gemini_json(full_prompt)
        # minimal validation
        if not isinstance(data, dict):
            return None
        bins = data.get("gl_bins")
        msgs = data.get("gl_messages")
        base = data.get("gi_percent_base")
        if (isinstance(bins, list) and len(bins) == 3 and
            isinstance(msgs, list) and len(msgs) == 4 and
            isinstance(base, (int, float)) and base > 0):
            _DIABETES_CFG_CACHE = {"gl_bins": bins, "gl_messages": msgs, "gi_percent_base": float(base)}
            return _DIABETES_CFG_CACHE
    except Exception:
        pass
    return None

# ==========================
# Main
# ==========================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Real-time nutrition helper (agent-only nutrients)")
    parser.add_argument("--text", type=str, default=None, help="Run in text mode with the provided sentence (no audio)")
    parser.add_argument("--continuous", action="store_true", help="Keep listening for multiple questions (default: one-shot)")
    args = parser.parse_args()

    if not GEMINI_API_KEY:
        print("Missing GEMINI_API_KEY. Create a .env file with: GEMINI_API_KEY=your-api-key")
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
        # quick input check (optional: set VU_SECONDS in env to 0 to skip)
        seconds = float(os.getenv("VU_SECONDS", "3.0"))
        if seconds > 0:
            block_ms = float(os.getenv("VU_BLOCK_MS", "50"))
            print(f"Running VU meter on device [{selected}] for {seconds:.1f}s ...")
            with sd.InputStream(
                samplerate=SAMPLE_RATE,
                channels=1,
                device=selected,
                blocksize=int(SAMPLE_RATE * (block_ms / 1000.0)),
                dtype="float32",
            ) as stream:
                start = time.time()
                while time.time() - start < seconds:
                    audio, _ = stream.read(int(SAMPLE_RATE * (block_ms / 1000.0)))
                    audio = audio * PRE_GAIN
                    rms = float(np.sqrt(np.mean(np.square(audio))))
                    peak = float(np.max(np.abs(audio)))
                    print(f"\rRMS:{rms*32768:6.1f}  Peak:{peak*32768:6.1f}", end="", flush=True)
                print("\n")
    except Exception as e:
        print(f"VU meter error (continuing anyway): {e}")

    try:
        listen_and_answer(selected, agent, single_shot=(not args.continuous))
    except KeyboardInterrupt:
        print("\nExiting.")
