#!/usr/bin/env python3
# python face_classifier.py 0 --display
from person_db import Person
from person_db import Face
from person_db import PersonDB
import face_recognition
import face_recognition_models
import numpy as np
from datetime import datetime
import cv2
import dlib
import face_alignment_dlib


class FaceClassifier():
    def __init__(self, threshold, ratio):
        self.similarity_threshold = threshold
        self.ratio = ratio
        self.predictor = dlib.shape_predictor(face_recognition_models.pose_predictor_model_location())

    def mosaic(self, frame, box):
        """모자이크 처리"""
        (top, right, bottom, left) = box
        face_region = frame[top:bottom, left:right]
        face_region = cv2.resize(face_region, (10, 10), interpolation=cv2.INTER_LINEAR)
        face_region = cv2.resize(face_region, (right-left, bottom-top), interpolation=cv2.INTER_NEAREST)
        frame[top:bottom, left:right] = face_region

    def detect_faces(self, frame, person_db):
        boxes = self.locate_faces(frame)
        if len(boxes) == 0:
            return []

        faces = []
        for box in boxes:
            face_image = self.get_face_image(frame, box)
            aligned_image = face_alignment_dlib.get_aligned_face(self.predictor, face_image)

            height, width = aligned_image.shape[:2]
            x = int(width / 3)
            y = int(height / 3)
            box_of_face = (y, x*2, y*2, x)
            encoding = face_recognition.face_encodings(aligned_image, [box_of_face])[0]

            face = Face("", face_image, encoding)
            face.location = box

            # 첫 번째 사람인지 확인
            if person_db.is_first_person(face):
                face.name = person_db.first_person.name
            else:
                face.name = "unknown"
                # unknown 얼굴 모자이크 처리
                self.mosaic(frame, box)

            faces.append(face)
        return faces



if __name__ == '__main__':
    dir_name = "result"
    pdb = PersonDB()
    pdb.load_db(dir_name)
    pdb.print_persons()

    fc = FaceClassifier(threshold=0.44, ratio=1.0)
    src = cv2.VideoCapture(0)  # 웹캠 사용

    while True:
        ret, frame = src.read()
        if not ret:
            break

        faces = fc.detect_faces(frame, pdb)

        cv2.imshow("Frame", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    src.release()
    cv2.destroyAllWindows()
