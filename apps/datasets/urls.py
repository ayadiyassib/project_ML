from django.urls import path

from apps.datasets import views

app_name = "datasets"

urlpatterns = [
    path("", views.index_view, name="index"),
    path("upload/", views.upload_view, name="upload"),
    path("sample/", views.sample_view, name="sample"),
    path("url/", views.url_view, name="url"),
    path("preview/", views.preview_view, name="preview"),
    path("activate/<uuid:pk>/", views.activate_view, name="activate"),
    path("delete/", views.delete_view, name="delete"),
]
