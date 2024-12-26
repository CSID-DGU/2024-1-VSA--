from django.conf import settings
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework.parsers import MultiPartParser, FormParser
from rest_framework.permissions import IsAuthenticated
from rest_framework import status
import cv2
import os
from tempfile import NamedTemporaryFile
from .utils import Mosaic, Get_normal_bbox
from .detection import initialize_models, Detection, extract_embeddings, Recognition
from .models import Video, FaceImage, FaceEmbedding
from .serializers import VideoSerializer, FaceImageSerializer
from PIL import Image
import numpy as np
import json
import requests
import torch

def download_file(url):
    response = requests.get(url, stream=True)
    if response.status_code == 200:
        temp_file = NamedTemporaryFile(delete=False, suffix=".mp4")
        with open(temp_file.name, 'wb') as f:
            for chunk in response.iter_content(1024):
                f.write(chunk)
        return temp_file.name
    else:
        raise Exception(f"Failed to download file from {url}")

class VideoBlurView(APIView):
    parser_classes = (MultiPartParser, FormParser)
    permission_classes = [IsAuthenticated]

    def post(self, request):
        # Parse input data
        video_serializer = VideoSerializer(data=request.data)
        if not video_serializer.is_valid():
            return Response(video_serializer.errors, status=400)
        video_instance = video_serializer.save(user=request.user)

        face_images = request.FILES.getlist('face_images')
        face_image_instances = []
        for img in face_images:
            face_image_serializer = FaceImageSerializer(data={"image_file": img})
            if face_image_serializer.is_valid():
                face_image = face_image_serializer.save(user=request.user)
                face_image_instances.append(face_image)
            else:
                return Response(face_image_serializer.errors, status=400)

        detector = request.data.get('detector', 'yolov8')  # Default to YOLOv8

        # Initialize models
        model_args = initialize_models(detector_type=detector)

        # Load face database
        face_db = self.load_face_database(request.user, model_args)

        # Download video file locally
        video_url = video_instance.video_file.url
        video_path = download_file(video_url)

        # Process video
        processed_video_path = self.process_video(video_path, face_image_instances, model_args, face_db)

        # Save processed video
        with open(processed_video_path, 'rb') as processed_file:
            video_instance.processed_video_file.save(
                os.path.basename(processed_video_path), processed_file
            )
        video_instance.status = 'Completed'
        video_instance.save()

        # Return processed video details
        return Response({
            "message": "Video processed successfully",
            "data": VideoSerializer(video_instance).data
        })

    def process_video(self, video_path, face_images, model_args, face_db):
        cap = cv2.VideoCapture(video_path)
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))

        processed_video_path = os.path.join(settings.MEDIA_ROOT, 'processed_videos', os.path.basename(video_path))
        os.makedirs(os.path.dirname(processed_video_path), exist_ok=True)
        out = cv2.VideoWriter(processed_video_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Detect faces
            bboxes = Detection(frame, model_args)
            bboxes = Get_normal_bbox(frame.shape, bboxes)

            # Recognize faces
            embeddings = extract_embeddings(frame, bboxes, model_args)
            face_ids = Recognition(embeddings, face_db)

            # Apply Mosaic (Blur)
            frame = Mosaic(frame, bboxes, face_ids)

            out.write(frame)

        cap.release()
        out.release()
        return processed_video_path

    def load_face_database(self, user, model_args):
        """
        Load face database from the database using JSON format.
        Includes the user's profile image if available.
        """
        face_db = {}

        # Load profile image embedding if available
        if user.profile_image:
            try:
                img_path = user.profile_image.path
                img = Image.open(img_path).convert('RGB')
                img = np.array(img)  # Convert to numpy array
                bboxes = Detection(img, model_args)
                embeddings = extract_embeddings(img, bboxes, model_args)
                if embeddings:
                    face_db['profile_image'] = embeddings.tolist()  # Store directly as list
            except Exception:
                pass

        # Load embeddings from FaceEmbedding model
        embeddings = FaceEmbedding.objects.filter(user=user)

        for embedding in embeddings:
            face_db[embedding.name] = torch.tensor(json.loads(embedding.embedding))  # JSON -> Tensor

        return face_db

class VideoListView(APIView):
    permission_classes = [IsAuthenticated]

    def get(self, request):
        videos = Video.objects.filter(user=request.user)
        serializer = VideoSerializer(videos, many=True)
        return Response({
            "message": "Video list fetched successfully",
            "data": serializer.data
        }, status=status.HTTP_200_OK)

class VideoDetailView(APIView):
    permission_classes = [IsAuthenticated]

    def get(self, request, pk):
        try:
            video = Video.objects.get(pk=pk, user=request.user)
            serializer = VideoSerializer(video)
            return Response({
                "message": "Video details fetched successfully",
                "data": serializer.data
            }, status=status.HTTP_200_OK)
        except Video.DoesNotExist:
            return Response({"error": "Video not found."}, status=status.HTTP_404_NOT_FOUND)