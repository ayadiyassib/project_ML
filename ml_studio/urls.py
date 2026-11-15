"""
URL configuration for ml_studio project.
"""

from django.contrib import admin
from django.urls import path, include
from django.conf import settings
from django.conf.urls.static import static

urlpatterns = [
    path('admin/', admin.site.urls),
    path('', include('apps.core.urls', namespace='core')),
    path('datasets/', include('apps.datasets.urls', namespace='datasets')),
    path('eda/', include('apps.eda.urls', namespace='eda')),
    path('preprocessing/', include('apps.preprocessing.urls', namespace='preprocessing')),
    path('models/', include('apps.ml_engine.urls', namespace='ml_engine')),
    path('diagnostics/', include('apps.model_viz.urls', namespace='model_viz')),
    path('export/', include('apps.export_inference.urls', namespace='export_inference')),
    path('docs/', include('apps.docs_tutorials.urls', namespace='docs_tutorials')),
]

if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
    urlpatterns += static(settings.STATIC_URL, document_root=settings.STATIC_ROOT)
