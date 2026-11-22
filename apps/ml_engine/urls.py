"""
URL patterns for ML Engine app.
"""

from django.urls import path
from apps.ml_engine import views

app_name = "ml_engine"

urlpatterns = [
    path("", views.index_view, name="index"),
    path("train/", views.train_view, name="train"),
    path("params/", views.params_view, name="params"),
    path("leaderboard/", views.leaderboard_view, name="leaderboard"),
    path("reset/", views.reset_view, name="reset"),
]
