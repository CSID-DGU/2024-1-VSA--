from ultralytics import YOLO
from django.conf import settings
import torch

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
        from facenet_pytorch import MTCNN
        models['Detection'] = MTCNN(device=device, keep_all=True)
    else:
        raise ValueError(f"Unsupported detector type: {detector_type}")

    # Initialize recognition model (FaceNet)
    from facenet_pytorch import InceptionResnetV1
    models['Recognition'] = InceptionResnetV1(pretrained='vggface2').eval().to(device)

    return models