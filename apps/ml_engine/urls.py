from django.urls import path
from apps.ml_engine import views

app_name = "ml_engine"

urlpatterns = [
    path("", views.index_view, name="index"),
]

