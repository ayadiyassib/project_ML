from django.urls import path
from apps.preprocessing import views

app_name = "preprocessing"

urlpatterns = [
    path("", views.index_view, name="index"),
]

