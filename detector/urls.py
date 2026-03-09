from django.urls import path

from .views import detect_size, detect_upload, index

urlpatterns = [
    path("", index, name="index"),
    path("detect/upload", detect_upload, name="detect_upload"),
    path("detect/size", detect_size, name="detect_size"),
]
