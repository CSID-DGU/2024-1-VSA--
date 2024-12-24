from time import time
import cv2
from PIL import Image
import torch
import torchvision
import numpy as np
from util import Mosaic, DrawRectImg
from detection import mtcnn_detection
from args import Args
from ultralytics import YOLO
import ml_part as ML
from detection import load_face_db
from facenet_pytorch import MTCNN, InceptionResnetV1

from retinaface_utils.utils.model_utils import load_model
from retinaface_utils.models.retinaface import RetinaFace
from retinaface_utils.data.config import cfg_mnet

# python main.py --video_path ../data/dest_images/ex2.mp4
def init(args):
    model_args = {}
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model_args['Device'] = device
    if args['DEBUG_MODE']:
        print('Running on device:', device)

    # Load Detection Model
    if args['DETECTOR'] == 'mtcnn':
        print("Using MTCNN for face detection.")
        mtcnn = MTCNN(image_size=160, margin=0, min_face_size=20, thresholds=[0.6, 0.7, 0.7],
                      factor=0.709, post_process=True, device=device, keep_all=True)
        model_args['Detection'] = mtcnn
        model_args['Mtcnn'] = mtcnn  # Add MTCNN to model_args for Recognition
    elif args['DETECTOR'] == 'yolov8':
        print("Using YOLOv8 for face detection.")
        yolov8_model = YOLO("yolov8m-face.pt")  # Pretrained YOLOv8 model
        model_args['Detection'] = yolov8_model
        # Add MTCNN for Recognition
        mtcnn = MTCNN(image_size=160, margin=0, min_face_size=20, thresholds=[0.6, 0.7, 0.7],
                      factor=0.709, post_process=True, device=device, keep_all=True)
        model_args['Mtcnn'] = mtcnn
    else:
        raise ValueError(f"Unknown detector: {args['DETECTOR']}")

    # Load FaceNet for Recognition
    resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)
    model_args['Recognition'] = resnet

    # Load Face Database
    face_db_path = "./database/face_db"
    face_db = load_face_db("../data/test_images", face_db_path, "./database/img_db", device, args, model_args)
    model_args['Face_db'] = face_db

    return model_args




def yolov8_detection(model, img):
    # Perform inference
    results = model(img)
    bboxes = []
    for box in results[0].boxes:  # Use .boxes to access bounding boxes
        x_min, y_min, x_max, y_max = box.xyxy[0].cpu().numpy()
        bboxes.append([x_min, y_min, x_max, y_max])
    return bboxes

import subprocess

def merge_audio_video(video_path, processed_video_path, save_path):
    try:
        command = [
            "ffmpeg",
            "-i", processed_video_path,
            "-i", video_path,
            "-c:v", "copy",
            "-c:a", "aac",
            "-map", "0:v:0",
            "-map", "1:a:0",
            save_path
        ]
        subprocess.run(command, check=True)
        print(f"Video saved with audio at: {save_path}")
    except Exception as e:
        print(f"Error during audio-video merging: {e}")

def ProcessImage(img, args, model_args):
    process_target = args['PROCESS_TARGET']
    detection_model = model_args['Detection']

    # Object Detection
    try:
        if args['DETECTOR'] == 'yolov8':
            bboxes = yolov8_detection(detection_model, img)
        elif args['DETECTOR'] == 'mtcnn':
            bboxes = mtcnn_detection(detection_model, img, model_args['Device'])
        else:
            raise ValueError(f"Unknown detector: {args['DETECTOR']}")
    except Exception as e:
        print(f"Error during detection: {e}")
        bboxes = []

    # Check if bboxes is empty or None
    if bboxes is None or len(bboxes) == 0:
        print("No valid bboxes detected.")
        if args['DETECTOR'] == 'mtcnn' and process_target == 'Video':
            if not isinstance(img, np.ndarray):  # NumPy 배열인지 확인
                img = img.numpy()
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        return img

    # Object Recognition
    try:
        face_ids = ML.Recognition(img, bboxes, args, model_args)
        print(f"Face IDs: {face_ids}")  # 디버깅 메시지 추가
    except Exception as e:
        print(f"Error during recognition: {e}")
        face_ids = ['unknown'] * len(bboxes)

    # Mosaic
    try:
        img = Mosaic(img, bboxes, face_ids, n=10)
    except Exception as e:
        print(f"Error during mosaic: {e}")

    # DrawRectImg
    try:
        processed_img = DrawRectImg(img, bboxes, face_ids)
    except Exception as e:
        print(f"Error during DrawRectImg: {e}")
        processed_img = img

    return processed_img


def main(args):
    model_args = init(args)

    # =================== Image =======================
    image_dir = args['IMAGE_DIR']
    if args['PROCESS_TARGET'] == 'Image':
        if args['DETECTOR'] in ['mtcnn', 'yolov8']:
            # Color channel: RGB
            if image_dir[-3:].upper() == 'PNG':
                img = Image.open(image_dir).convert('RGB')
            else:
                img = Image.open(image_dir)

            img = ProcessImage(img, args, model_args)

            cv2.imwrite(args['SAVE_DIR'] + '/output.jpg', img)
    # =================== Image =======================

    # =================== Video =======================
    elif args['PROCESS_TARGET'] == 'Video':
        video_path = args['VIDEO_PATH']
        processed_video_path = args['SAVE_DIR'] + '/processed_video.mp4'
        save_path = args['SAVE_DIR'] + '/output_with_audio.mp4'

        if args['PROCESS_TARGET'] == 'Video':
            cap = cv2.VideoCapture(video_path)
            frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = int(cap.get(cv2.CAP_PROP_FPS))

            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(processed_video_path, fourcc, fps, (frame_width, frame_height))

            while True:
                ret, img = cap.read()
                if not ret:
                    break
                img = ProcessImage(img, args, model_args)
                out.write(img)

            cap.release()
            out.release()

            merge_audio_video(video_path, processed_video_path, save_path)
    # ====================== Video ===========================



if __name__ == "__main__":
    args = Args().params
    main(args)
