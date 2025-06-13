"""
API URLs for ML Manager.
"""

from django.urls import path, include
from rest_framework.routers import DefaultRouter
from .views import MLModelViewSet, PredictionViewSet

router = DefaultRouter()
router.register(r'models', MLModelViewSet)
router.register(r'predictions', PredictionViewSet)

urlpatterns = [
    path('', include(router.urls)),
]
