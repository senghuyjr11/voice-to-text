# voice_to_text.py
# Real-time headset-mic dictation → Agent-only food extraction + nutrients (no local CSV) → Diabetes metrics → JSON log
# - Locks to a headset/earphone mic (auto-detect; env override supported)
# - Transcribes in near-real-time with faster-whisper
# - Uses Gemini to return a STRUCTURED JSON payload with foods, portions, and nutrients (per 100g and/or per portion)
# - Computes net carbs and glycemic load (GL) from the agent’s data
# - Saves full structured result to ./answers/answer_YYYYmmdd_HHMMSS.json
# - Single-shot by default (one question → one answer → exit). Use --continuous for ongoing.
# - Extras: VU meter clipping hint, robust JSON extraction, Gemini timeout/retries, optional --text mode

import os
import sys
import time
import json
import queue
import argparse
import numpy as np
import sounddevice as sd
from collections import deque
from datetime import datetime
from dotenv import load_dotenv
from faster_whisper import WhisperModel
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeout
from typing import Optional

# --------------------------
# Environment & Config
# --------------------------
load_dotenv()

SAMPLE_RATE = int(os.getenv("SAMPLE_RATE", "16000"))
CHANNELS = 1

# Audio device selection
INPUT_DEVICE_INDEX_ENV = os.getenv("INPUT_DEVICE_INDEX")  # optional manual override
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
    Two-step agent:
      1) Extract foods and portions from free text → JSON.
      2) For any food missing nutrients, call the model again to fill:
         - per_100g: carbs_g, fiber_g, sugars_g, protein_g, fat_g, calories
         - per_portion: grams and the same nutrients if known
         - gi: value if known (otherwise omit or give estimate with 'confidence')
    Returns a single JSON object with foods fully enriched when possible.
    """

    SCHEMA_EXTRACT = {
        "type": "object",
        "properties": {
            "intent": {"enum": ["can_i_eat", "nutrition_question", "unknown"]},
            "foods": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "brand": {"type": "string"},
                        "quantity": {
                            "type": "object",
                            "properties": {
                                "amount": {"type": "number"},
                                "unit": {"type": "string"},
                            },
                        },
                        "size": {"type": "string"},
                        "cooking_method": {"type": "string"},
                        "modifiers": {"type": "array", "items": {"type": "string"}},
                    },
                    "required": ["name"],
                },
            },
        },
        "required": ["foods"],
    }

    SCHEMA_ENRICH = {
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "portion_grams": {"type": "number"},
            "per_100g": {
                "type": "object",
                "properties": {
                    "calories": {"type": "number"},
                    "carbs_g": {"type": "number"},
                    "fiber_g": {"type": "number"},
                    "sugars_g": {"type": "number"},
                    "protein_g": {"type": "number"},
                    "fat_g": {"type": "number"},
                },
                "required": ["carbs_g"],
            },
            "per_portion": {
                "type": "object",
                "properties": {
                    "grams": {"type": "number"},
                    "calories": {"type": "number"},
                    "carbs_g": {"type": "number"},
                    "fiber_g": {"type": "number"},
                    "sugars_g": {"type": "number"},
                    "protein_g": {"type": "number"},
                    "fat_g": {"type": "number"},
                },
            },
            "gi": {
                "type": "object",
                "properties": {
                    "value": {"type": "number"},
                    "source": {"type": "string"},
                    "confidence": {"enum": ["low", "medium", "high"]},
                },
            },
        },
        "required": ["name", "per_100g"],
    }

    def __init__(self, api_key: Optional[str], model_name: str):
        self.model = None
        self.genai = None
        if api_key:
            try:
                import google.generativeai as genai
                genai.configure(api_key=api_key)
                self.genai = genai
                self.model = genai.GenerativeModel(model_name)
            except Exception as e:
                print(f"Gemini init warning: {e}")
                self.model = None

        self._pool = ThreadPoolExecutor(max_workers=2)

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

    # ---- step 1: extraction
    def _extract_only(self, question: str) -> Optional[dict]:
        if not self.model:
            return None

        system = (
            "Extract foods and amounts from the user's sentence. "
            "Return ONLY valid JSON matching this SCHEMA (no prose). "
            "Be concise and do not invent foods not mentioned.\n\n"
            f"SCHEMA:\n{json.dumps(self.SCHEMA_EXTRACT)}\n\n"
            "EXAMPLES:\n"
            "USER: Can I eat a bowl of white rice with fried chicken?\n"
            'JSON: {"intent":"can_i_eat","foods":[{"name":"white rice","quantity":{"amount":1,"unit":"bowl"}},{"name":"fried chicken"}]}\n'
            "USER: Is a medium banana okay before the gym?\n"
            'JSON: {"intent":"can_i_eat","foods":[{"name":"banana","size":"medium"}]}\n'
        )
        prompt = system + "\nUSER:\n" + question + "\nJSON:"
        return self._call_gemini_json(prompt)

    # ---- step 2: enrichment for a single food
    def _enrich_food(self, name: str, quantity: Optional[dict], size: Optional[str], cooking_method: Optional[str]) -> Optional[dict]:
        if not self.model:
            return None

        qty_str = ""
        if quantity and (quantity.get("amount") or quantity.get("unit")):
            qty_str = f' Given portion: amount={quantity.get("amount")}, unit="{quantity.get("unit")}".'
        if size:
            qty_str += f' Size hint: "{size}".'
        if cooking_method:
            qty_str += f' Method: "{cooking_method}".'

        system = (
            "For the named food, return ONLY JSON with per-100g nutrients (required), a realistic portion in grams, "
            "optional per-portion nutrients, and GI if known. "
            "Use grams for weights. If GI unknown, omit it. Do not include prose.\n\n"
            f"SCHEMA:\n{json.dumps(self.SCHEMA_ENRICH)}\n\n"
            "EXAMPLES:\n"
            'FOOD: "fried chicken"\n'
            'JSON: {"name":"fried chicken","portion_grams":150,'
            '"per_100g":{"calories":246,"carbs_g":8.0,"fiber_g":0.3,"sugars_g":0.2,"protein_g":23.0,"fat_g":13.0},'
            '"per_portion":{"grams":150,"calories":369,"carbs_g":12.0,"fiber_g":0.5,"sugars_g":0.3,"protein_g":34.5,"fat_g":19.5}}\n'
            'FOOD: "white rice (cooked)"\n'
            'JSON: {"name":"white rice (cooked)","portion_grams":150,'
            '"per_100g":{"calories":130,"carbs_g":28.0,"fiber_g":0.4,"sugars_g":0.1,"protein_g":2.4,"fat_g":0.3},'
            '"per_portion":{"grams":150,"calories":195,"carbs_g":42.0,"fiber_g":0.6,"sugars_g":0.2,"protein_g":3.6,"fat_g":0.5},'
            '"gi":{"value":73,"confidence":"medium"}}\n'
        )

        user = f'FOOD: "{name}".{qty_str}\nJSON:'
        prompt = system + "\n" + user
        return self._call_gemini_json(prompt)

    def _ensure_enriched_food(self, food: dict) -> Optional[dict]:
        name = (food.get("name") or "").strip()
        if not name:
            return None
        quantity = food.get("quantity") or {}
        size = food.get("size")
        method = food.get("cooking_method")

        enriched = self._enrich_food(name, quantity, size, method)
        if not enriched:
            return None

        merged = {
            "name": enriched.get("name", name),
            "brand": food.get("brand"),
            "quantity": quantity if quantity else None,
            "size": size,
            "cooking_method": method,
            "portion_grams": enriched.get("portion_grams"),
            "per_100g": enriched.get("per_100g"),
            "per_portion": enriched.get("per_portion"),
            "gi": enriched.get("gi"),
        }
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


# --------------------------
# Diabetes Metrics from Agent JSON
# --------------------------

def safe_num(x, default=None):
    try:
        return float(x)
    except Exception:
        return default


def pick_portion_grams(food: dict) -> Optional[float]:
    per_portion = food.get("per_portion") or {}
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
    pp = food.get("per_portion") or {}
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


def _diabetes_guidance_lines(gi_known: bool) -> list[str]:
    lines = []
    if gi_known:
        lines.append("*   **Diabetes:** GI/GL available. Use GL to judge: ≤10 low, 11–19 moderate, ≥20 high.")
    else:
        lines.append("*   **Diabetes:** GI not provided for at least one item. Use **net carbs** (carbs − fiber) and portion control.")
    lines.append("*   **Portion tips:** Start with ~15–30 g net carbs per snack or ~45–60 g per meal (individualize per clinician).")
    lines.append("*   **Pairing:** Combine carbs with protein or healthy fats (e.g., nuts, yogurt, eggs) to slow glucose rise.")
    lines.append("*   **Ripeness/cooking:** Riper fruit & longer-cooked starches → higher GI; less ripe/al dente → lower GI.")
    lines.append("*   **Monitor:** Check fingerstick/CGM at ~1–2 hours post-meal; adjust next time.")
    lines.append("*   **Red flags:** If advised to limit potassium (kidney issues), watch high-K foods (e.g., bananas).")
    return lines


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

    # Health implications with diabetes-specific guidance
    lines.append("**Health Implications:**")
    gi_known = any(it.get("gi") is not None for it in items)
    lines.extend(_diabetes_guidance_lines(gi_known))
    lines.append("*   **Diet:** Fit within your daily goals; fiber and protein support satiety.")
    lines.append("*   **General:** Hydrate; consider activity (walk ~10–15 min post meal) to lower glucose.")
    lines.append("")

    # Practical recs
    lines.append("**Practical Recommendations:**")
    lines.append("*   **Moderation:** Keep portions sensible; adjust based on readings and activity.")
    lines.append("*   **Swap-smart:** Prefer lower-GI options when possible (e.g., berries over juice; brown rice over sticky white rice).")
    lines.append("*   **Timing:** If using insulin/meds, align timing with carb load per your care plan.")
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