from django.urls import path
from apps.model_viz import views

app_name = "model_viz"

urlpatterns = [
    path("", views.index_view, name="index"),
]

