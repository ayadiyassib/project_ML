from django.urls import path

from apps.eda import views

app_name = "eda"

urlpatterns = [
    path("", views.index_view, name="index"),
    path("univariate/", views.univariate_view, name="univariate"),
    path("correlation/", views.correlation_view, name="correlation"),
    path("scatter/", views.scatter_view, name="scatter"),
    path("scatter3d/", views.scatter3d_view, name="scatter3d"),
    path("pairplot/", views.pairplot_view, name="pairplot"),
]
