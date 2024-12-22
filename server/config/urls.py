from django.contrib import admin
from django.conf import settings
from django.urls import path, include
from rest_framework.routers import DefaultRouter

router = DefaultRouter()

urlpatterns = [
    path('admin/', admin.site.urls),
    path('api/', include(router.urls)),
    path('api/auth/', include('accounts.urls')),  # accounts 관련
    path('api/video/', include('accounts.urls')), # video 관련
]