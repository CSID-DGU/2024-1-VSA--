import os
import json
from PIL import Image
import numpy as np
from facenet_pytorch import MTCNN, InceptionResnetV1
from ultralytics import YOLO
import torch
from django.conf import settings
from .models import FaceEmbedding

def initialize_models(detector_type='yolov8', device=None):
    """
    Initialize detection and recognition models based on settings.
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    models = {'Device': device}

    if detector_type == 'yolov8':
        model_path = settings.YOLO_MODEL_PATH
        models['Detection'] = YOLO(model_path)
    elif detector_type == 'mtcnn':
        models['Detection'] = MTCNN(device=device, keep_all=True)
    else:
        raise ValueError(f"Unsupported detector type: {detector_type}")

    models['Recognition'] = InceptionResnetV1(pretrained='vggface2').eval().to(device)
    return models

def Detection(img, model_args, detector_type='yolov8'):
    """
    Detect faces using YOLOv8 or MTCNN.
    """
    detector = model_args['Detection']
    if detector_type == 'yolov8':
        results = detector(img)
        bboxes = results[0].boxes.xyxy.cpu().numpy() if results else []
    elif detector_type == 'mtcnn':
        bboxes, _ = detector.detect(img)
        bboxes = bboxes if bboxes is not None else []
    else:
        bboxes = []
    return bboxes

def extract_embeddings(img, bboxes, model_args):
    """
    Extract face embeddings using bounding boxes and InceptionResnetV1.
    """
    resnet = model_args['Recognition']
    device = model_args['Device']

    faces = []
    for bbox in bboxes:
        x1, y1, x2, y2 = [int(coord) for coord in bbox]
        face = img[y1:y2, x1:x2]  # Crop the face
        face = Image.fromarray(face).resize((160, 160))  # Resize to 160x160 for ResNet
        face = np.array(face).astype(np.float32) / 255.0  # Normalize pixel values
        face = torch.tensor(face).permute(2, 0, 1).unsqueeze(0).to(device)  # Convert to tensor

        faces.append(face)

    if faces:
        faces = torch.cat(faces)  # Combine all face tensors into a batch
        embeddings = resnet(faces).detach().cpu()
        return embeddings
    return []

def build_face_database(user, model_args):
    """
    Build a database of known face embeddings for a user.
    """
    face_db = {}
    face_images = user.faceimage_set.all()  # Assuming related name for FaceImage is `faceimage_set`

    for face_image in face_images:
        try:
            img_path = face_image.image_file.path
            img = Image.open(img_path).convert('RGB')
            img = np.array(img)  # Convert to numpy array
            bboxes, _ = model_args['Detection'].detect(img)
            if bboxes is not None:
                embeddings = extract_embeddings(img, bboxes, model_args)
                if embeddings:
                    face_db[face_image.id] = embeddings.tolist()  # Convert tensor to list
        except Exception as e:
            print(f"Error processing {face_image.image_file.name}: {e}")

    # Save to FaceEmbedding
    for name, embedding in face_db.items():
        FaceEmbedding.objects.create(
            user=user,
            name=name,
            embedding=json.dumps(embedding)  # Save embedding as JSON string
        )

    return face_db

def load_face_database(user):
    """
    Load face database from the database using JSON format.
    """
    face_db = {}
    embeddings = FaceEmbedding.objects.filter(user=user)

    for embedding in embeddings:
        face_db[embedding.name] = torch.tensor(json.loads(embedding.embedding))  # JSON -> Tensor

    return face_db

def Recognition(embeddings, face_db, threshold=0.85, metric='euclidean'):
    """
    Recognize faces by comparing embeddings with the face database.
    """
    recognized_faces = []
    for emb in embeddings:
        best_match = 'unknown'
        best_score = float('inf') if metric == 'euclidean' else -float('inf')

        for name, known_embeddings in face_db.items():
            for known_emb in known_embeddings:
                if metric == 'euclidean':
                    score = torch.norm(emb - known_emb).item()
                    if score < best_score:
                        best_match = name
                        best_score = score
                elif metric == 'cosine':
                    score = torch.nn.functional.cosine_similarity(emb, known_emb.unsqueeze(0)).item()
                    if score > best_score:
                        best_match = name
                        best_score = score

        if metric == 'euclidean' and best_score < threshold:
            recognized_faces.append(best_match)
        elif metric == 'cosine' and best_score > threshold:
            recognized_faces.append(best_match)
        else:
            recognized_faces.append('unknown')

    return recognized_faces