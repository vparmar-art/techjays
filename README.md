# HVAC Duct Detector (Django Assignment App)

This project is a Django web app for HVAC drawing analysis.

Current scope:
- Upload one mechanical drawing PDF.
- Run LSD geometry extraction once.
- Let the user choose a duct diameter (`8`, `10`, `12`, `14`, `18` inches).
- Run OCR for the selected diameter and highlight matching duct boxes.

## Current Implementation Approach

### Two-step pipeline

#### Step A: Upload + LSD pre-processing (`POST /detect/upload`)
- Render first PDF page.
- Estimate plan ROI.
- Run tiled OpenCV LSD on ROI.
- Build LSD box candidates from axis-aligned line structure.

#### Step B: Diameter selection + OCR matching (`POST /detect/size`)
- Reuse cached LSD output from Step A (LSD does **not** rerun).
- Run Google Vision OCR on the plan crop for the chosen diameter.
- Normalize/clean marker detections (`8`, `10`, `12`, `14`, `18` inch formats).
- Match marker-to-LSD boxes (text-in-box / overlap logic).
- Draw overlay:
  - blue = matched duct boxes,
  - red = detected text markers.

## Tech Stack

- Python 3
- Django 6
- OpenCV (LSD)
- PyMuPDF (`fitz`) for PDF rasterization
- NumPy
- Google Vision API (via key-based HTTP request)

## Project Structure

- `detector/duct_detection.py`
  - Core detection pipeline, OCR parsing, matching, overlay generation
- `detector/views.py`
  - Upload and size endpoints
- `detector/templates/detector/index.html`
  - UI and AJAX behavior
- `media/runs/<run_id>/`
  - Per-run debug artifacts and outputs

## Known Limits (Current MVP)

- Works on first page only.
- Marker-driven matching can miss ducts if OCR misses labels.
- Focus is box-level highlighting, not full connected duct tracing.
- No production hardening (auth, queueing, autoscaling, SLOs).

## Possible Future Improvements

### Better duct detection quality
- Add OCR calibration pass per run:
  - detect and correct marker offset/drift before matching.
- Move from local box match to connected duct component extraction:
  - seed from marker location,
  - traverse line graph through bends/elbows,
  - return full duct boundary paths.
- Add multi-signal fusion:
  - LSD + contour/edge maps + text proximity + topology rules.
- Improve size parsing coverage:
  - handle more notation variants and noisy OCR token splits.
- Add confidence-driven filtering and operator review mode:
  - accept high-confidence auto matches,
  - flag low-confidence candidates for click-to-confirm.
- Add support for rectangular duct labels (`14x10`, etc.) and pressure class extraction.

### Performance and reliability
- Cache OCR results by `(run_id, diameter)` to avoid repeat API calls.
- Background processing with Celery/RQ for large drawings.
- Add request-level tracing and structured logs for failure diagnostics.

### Hosting / deployment
- Containerize app with Docker.
- Deploy to a managed platform:
  - Render / Railway / Fly.io for quick assignment demo,
  - or ECS/GKE/App Runner for scale.
- Use production web stack:
  - Gunicorn + Nginx.
- Use cloud object storage for artifacts:
  - S3/GCS instead of local `media/`.
- Add production settings:
  - `DEBUG=false`, secure secrets, HTTPS, host allowlist, error monitoring.
- Add simple CI pipeline:
  - dependency install,
  - Django checks,
  - lint/format gate,
  - deploy on main branch.
