from django.urls import path

from apps.model_viz import views

app_name = "model_viz"

urlpatterns = [
    path("", views.index_view, name="index"),
    path("boundary/", views.boundary_view, name="boundary"),
    path("permutation/", views.permutation_view, name="permutation"),
    path("elbow/", views.elbow_view, name="elbow"),
]
