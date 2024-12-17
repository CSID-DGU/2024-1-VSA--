from django.urls import path
from .views import index
from .views import upload_video, download_video, getVideoList

urlpatterns = [
    path("", index),
    
    path("api/upload", upload_video),
    path('api/download/<str:filename>/', download_video, name='download_video'), 
    path("api/videos", getVideoList)
]