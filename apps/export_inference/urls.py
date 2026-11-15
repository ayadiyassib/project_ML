from django.urls import path
from apps.export_inference import views

app_name = "export_inference"

urlpatterns = [
    path("", views.index_view, name="index"),
]

