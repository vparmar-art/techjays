from django import forms


class PDFUploadForm(forms.Form):
    pdf_file = forms.FileField(
        label="HVAC Mechanical Drawing (PDF)",
        help_text="Upload the drawing PDF you want to analyze.",
    )
