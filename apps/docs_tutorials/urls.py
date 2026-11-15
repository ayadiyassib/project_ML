from django.urls import path
from apps.docs_tutorials import views

app_name = "docs_tutorials"

urlpatterns = [
    path("", views.index_view, name="index"),
    path("algorithm/<slug:slug>/", views.algorithm_detail_view, name="algorithm_detail"),
    path("tutorial/<slug:slug>/", views.tutorial_detail_view, name="tutorial_detail"),
]
