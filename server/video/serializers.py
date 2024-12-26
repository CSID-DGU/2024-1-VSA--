from rest_framework import serializers
from .models import Video, FaceImage, FaceEmbedding

class VideoSerializer(serializers.ModelSerializer):
    class Meta:
        model = Video
        fields = ('id', 'video_file', 'processed_video_file', 'status', 'created_at')

class FaceImageSerializer(serializers.ModelSerializer):
    class Meta:
        model = FaceImage
        fields = ('id', 'image_file', 'created_at')

class FaceEmbeddingSerializer(serializers.ModelSerializer):
    class Meta:
        model = FaceEmbedding
        fields = ('id', 'user', 'name', 'embedding', 'created_at')