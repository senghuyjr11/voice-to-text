# Real-Time Nutrition GL Helper

A CLI tool that listens to your headset mic, transcribes in near real-time, asks Gemini to extract foods and nutrients, computes net carbs and glycemic load (GL), and saves everything as structured JSON.

---

## Features
- Locks to headset/earphone mic (auto-detect; env override supported).
- Uses faster-whisper for near real-time transcription.
- Sends transcript to Gemini, which returns structured JSON:
  - foods, portions, nutrients (per 100g and/or per portion), optional GI.
- Computes net carbs and glycemic load (GL) per item and per meal.
- Saves results to `./answers/answer_YYYYmmdd_HHMMSS.json`.
- Single-shot by default; `--continuous` for ongoing listening.
- `--text "..."` for testing without audio.

---

## Installation

Make sure you have **Python 3.9+**, a headset or earphone mic, and optionally a CUDA GPU for faster-whisper acceleration.  

Create and activate a virtual environment, install dependencies, and configure `.env`:

```bash
# Create virtual environment
python -m venv .venv
source .venv/bin/activate   # Linux / Mac
.venv\Scripts\activate      # Windows

# Install dependencies
pip install -U faster-whisper sounddevice numpy python-dotenv google-generativeai

# Create a .env file in the project root with the following:
GEMINI_API_KEY=api_key