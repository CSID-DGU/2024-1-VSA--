from django.urls import path
from .views import VideoBlurView, VideoListView, VideoDetailView

urlpatterns = [
    path('', VideoListView.as_view(), name='video_list'),
    path('<int:pk>/', VideoDetailView.as_view(), name='video_detail'),
    path('video-blur/', VideoBlurView.as_view(), name='video_blur'),
]
