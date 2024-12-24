from django.urls import path
from .views import index
from .views import upload_video, download_video, getVideoList, login_view

urlpatterns = [
    path("", index),
    
    path("api/upload", upload_video),
    path('api/download/<str:filename>/', download_video, name='download_video'), 
    path("api/videos", getVideoList),
    path('api/auth/register', login_view, name='login'),
]