from django.urls import path
from apps.core import views

app_name = "core"

urlpatterns = [
    path("", views.dashboard_view, name="dashboard"),
    path("reset-session/", views.reset_session_view, name="reset_session"),
]

