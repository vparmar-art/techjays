import json
from pathlib import Path

from django.conf import settings
from django.http import JsonResponse
from django.shortcuts import render
from django.views.decorators.http import require_POST

from .duct_detection import SUPPORTED_DIAMETERS, detect_diameter_for_run, initialize_lsd_upload_run
from .forms import PDFUploadForm


def index(request):
    form = PDFUploadForm()
    return render(
        request,
        "detector/index.html",
        {
            "form": form,
            "media_url": settings.MEDIA_URL,
            "available_sizes": SUPPORTED_DIAMETERS,
        },
    )


@require_POST
def detect_upload(request):
    form = PDFUploadForm(request.POST, request.FILES)
    if not form.is_valid():
        return JsonResponse({"status": "error", "error": "Invalid form submission. Please upload a valid PDF file."}, status=400)

    try:
        pdf_file = form.cleaned_data["pdf_file"]
        pdf_bytes = pdf_file.read()
        payload = initialize_lsd_upload_run(pdf_bytes, Path(settings.MEDIA_ROOT))
    except Exception as exc:
        return JsonResponse({"status": "error", "error": str(exc)}, status=500)

    return JsonResponse(payload)


@require_POST
def detect_size(request):
    provider = str(getattr(settings, "AI_PROVIDER", "gemini")).strip().lower() or "gemini"

    try:
        if request.content_type and "application/json" in request.content_type.lower():
            payload = json.loads(request.body.decode("utf-8") or "{}")
        else:
            payload = request.POST
    except Exception:
        return JsonResponse({"status": "error", "error": "Invalid JSON payload."}, status=400)

    run_id = str(payload.get("run_id", "")).strip()
    diameter = payload.get("diameter")
    if not run_id:
        return JsonResponse({"status": "error", "error": "run_id is required."}, status=400)
    if diameter in (None, ""):
        return JsonResponse({"status": "error", "error": "diameter is required."}, status=400)

    try:
        result = detect_diameter_for_run(run_id, diameter, Path(settings.MEDIA_ROOT), provider)
    except ValueError as exc:
        return JsonResponse({"status": "error", "error": str(exc)}, status=400)
    except FileNotFoundError as exc:
        return JsonResponse({"status": "error", "error": str(exc)}, status=404)
    except Exception as exc:
        return JsonResponse({"status": "error", "error": str(exc)}, status=500)

    return JsonResponse(
        {
            "status": "ok",
            "run_id": result.get("run_id"),
            "selected_diameter": result.get("selected_diameter"),
            "overlay_url": result.get("files", {}).get("overlay"),
            "result_json_url": result.get("files", {}).get("json"),
            "summary": {
                "provider_used": result.get("provider_used"),
                "selection_mode": result.get("selection_mode"),
                "size_marker_count": result.get("size_marker_count", 0),
                "selected_candidate_count": result.get("selected_candidate_count", 0),
                "line_segment_count": result.get("line_segment_count", 0),
                "verification_status": result.get("verification", {}).get("status"),
                "strong_evidence_ratio": result.get("verification", {}).get("metrics", {}).get("strong_evidence_ratio"),
                "marker_covered_ratio": result.get("verification", {}).get("metrics", {}).get("marker_covered_ratio"),
                "boundary_complete_ratio": result.get("verification", {}).get("metrics", {}).get("boundary_complete_ratio"),
            },
            "warnings": result.get("warnings", []),
            "table_rows": result.get("duct_lines", []),
            "files": result.get("files", {}),
            "result": result,
        }
    )
