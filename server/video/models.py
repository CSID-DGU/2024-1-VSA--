from django.conf import settings
from django.db import models

class Video(models.Model):
    user = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.CASCADE)
    video_file = models.FileField(upload_to='videos/')
    processed_video_file = models.FileField(upload_to='processed_videos/', null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    status = models.CharField(max_length=20, default='Pending')

class FaceImage(models.Model):
    user = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.CASCADE)
    image_file = models.ImageField(upload_to='faces/')
    created_at = models.DateTimeField(auto_now_add=True)

class FaceEmbedding(models.Model):
    user = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.CASCADE)
    name = models.CharField(max_length=255)
    embedding = models.JSONField()
    created_at = models.DateTimeField(auto_now_add=True)