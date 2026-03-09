from pathlib import Path

from django.conf import settings
from django.shortcuts import render

from .duct_detection import run_detection
from .forms import PDFUploadForm


def index(request):
    result = None
    error = None

    if request.method == "POST":
        form = PDFUploadForm(request.POST, request.FILES)
        if form.is_valid():
            try:
                pdf_file = form.cleaned_data["pdf_file"]
                provider = str(getattr(settings, "AI_PROVIDER", "gemini")).strip().lower() or "gemini"
                pdf_bytes = pdf_file.read()
                result = run_detection(pdf_bytes, provider, Path(settings.MEDIA_ROOT))
            except Exception as exc:
                error = str(exc)
        else:
            error = "Invalid form submission. Please upload a valid PDF file."
    else:
        form = PDFUploadForm()

    return render(
        request,
        "detector/index.html",
        {
            "form": form,
            "result": result,
            "error": error,
            "media_url": settings.MEDIA_URL,
        },
    )
