"""
URL configuration for coronary_experiments project.

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/4.2/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.urls import path, include
from django.shortcuts import redirect
from django.contrib.auth import views as auth_views
from django.conf import settings
from django.conf.urls.static import static
from django.http import HttpResponse
from django.views.generic import RedirectView

def redirect_to_ml(request):
    return redirect('ml_manager:model-list')

def favicon_view(request):
    return HttpResponse(status=204)  # No Content for favicon

urlpatterns = [
    path('', redirect_to_ml, name='home'),
    path('admin/', admin.site.urls),
    path('ml/', include(('core.apps.ml_manager.urls', 'ml_manager'), namespace='ml_manager')),
    path('datasets/', include(('core.apps.dataset_manager.urls', 'dataset_manager'), namespace='dataset_manager')),
    
    # API endpoints
    path('api/ml/', include(('core.apps.ml_manager.api_urls', 'ml_manager_api'), namespace='ml_manager_api')),
    
    # Authentication
    path('accounts/login/', auth_views.LoginView.as_view(template_name='ml_manager/login.html'), name='login'),
    path('accounts/logout/', auth_views.LogoutView.as_view(next_page='home'), name='logout'),
    
    # Favicon
    path('favicon.ico', favicon_view, name='favicon'),
]

# Serve media files in development
if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
