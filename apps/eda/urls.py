from django.urls import path
from apps.eda import views

app_name = "eda"

urlpatterns = [
    path("", views.index_view, name="index"),
]

