# HVAC Duct Detector (Django Assignment MVP)

This is a minimal Django app for detecting ducts from HVAC mechanical drawing PDFs.

## What it does

- Upload a PDF drawing.
- Runs AI-assisted duct detection using exactly one provider (`gemini` or `openai`).
- No local fallback is used.
- Returns:
  - Annotated overlay image,
  - JSON file with normalized duct paths (`duct_segments`) and pixel line segments (`line_segments`),
  - Verification metrics (`good` / `warn` / `bad`) and parse warnings,
  - Raw provider response artifacts for debugging,
  - Run metadata for the selected provider.

## Setup

```bash
python3 -m venv venv
./venv/bin/pip install -r requirements.txt
./venv/bin/python manage.py migrate
```

## Configure `.env`

Edit `.env` in the project root and set:

- `AI_PROVIDER=gemini` or `AI_PROVIDER=openai`
- `AI_REQUEST_TIMEOUT_SECONDS` (read timeout, default `90`)
- `AI_CONNECT_TIMEOUT_SECONDS` (connect timeout, default `15`)
- `GEMINI_API_KEY` for Gemini runs
- `OPENAI_API_KEY` for OpenAI runs

## Run

```bash
./venv/bin/python manage.py runserver
```

Open `http://127.0.0.1:8000/` and upload your PDF.

Choose the same provider in the UI as your `AI_PROVIDER` value (or just keep the default).

## Output files

Each run writes to `media/runs/<run_id>/`:

- `page.png`
- `plan_crop.png`
- `overlay.png`
- `result.json`
- `provider_raw_response.json`
- `provider_parsed.json`
- `provider_model_text.txt`
