from django.urls import path
from apps.datasets import views

app_name = "datasets"

urlpatterns = [
    path("", views.index_view, name="index"),
]

