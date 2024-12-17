#!/usr/bin/env python3
# python face_classifier.py 0 -p ./pre_face --display

from person_db import Person
from person_db import Face
from person_db import PersonDB
import face_recognition
import numpy as np
from datetime import datetime
import cv2


class FaceClassifier():
    def __init__(self, threshold, ratio):
        self.similarity_threshold = threshold
        self.ratio = ratio

    def get_face_image(self, frame, box):
        img_height, img_width = frame.shape[:2]
        (box_top, box_right, box_bottom, box_left) = box
        box_width = box_right - box_left
        box_height = box_bottom - box_top
        crop_top = max(box_top - box_height, 0)
        pad_top = -min(box_top - box_height, 0)
        crop_bottom = min(box_bottom + box_height, img_height - 1)
        pad_bottom = max(box_bottom + box_height - img_height, 0)
        crop_left = max(box_left - box_width, 0)
        pad_left = -min(box_left - box_width, 0)
        crop_right = min(box_right + box_width, img_width - 1)
        pad_right = max(box_right + box_width - img_width, 0)
        face_image = frame[crop_top:crop_bottom, crop_left:crop_right]
        if (pad_top == 0 and pad_bottom == 0):
            if (pad_left == 0 and pad_right == 0):
                return face_image
        padded = cv2.copyMakeBorder(face_image, pad_top, pad_bottom,
                                    pad_left, pad_right, cv2.BORDER_CONSTANT)
        return padded

    # return list of dlib.rectangle
    def locate_faces(self, frame):
        #start_time = time.time()
        if self.ratio == 1.0:
            rgb = frame[:, :, ::-1]
        else:
            small_frame = cv2.resize(frame, (0, 0), fx=self.ratio, fy=self.ratio)
            rgb = small_frame[:, :, ::-1]
        boxes = face_recognition.face_locations(rgb)
        #elapsed_time = time.time() - start_time
        #print("locate_faces takes %.3f seconds" % elapsed_time)
        if self.ratio == 1.0:
            return boxes
        boxes_org_size = []
        for box in boxes:
            (top, right, bottom, left) = box
            left = int(left / ratio)
            right = int(right / ratio)
            top = int(top / ratio)
            bottom = int(bottom / ratio)
            box_org_size = (top, right, bottom, left)
            boxes_org_size.append(box_org_size)
        return boxes_org_size

    def detect_faces(self, frame):
        boxes = self.locate_faces(frame)
        if len(boxes) == 0:
            return []

        # faces found
        faces = []
        now = datetime.now()
        str_ms = now.strftime('%Y%m%d_%H%M%S.%f')[:-3] + '-'
        encodings = face_recognition.face_encodings(frame, boxes)
        for i, box in enumerate(boxes):
            face_image = self.get_face_image(frame, box)
            face = Face(str_ms + str(i) + ".png", face_image, encodings[i])
            face.location = box
            faces.append(face)
        return faces

    def compare_with_known_persons(self, face, persons):
        if len(persons) == 0:
            return None

        # see if the face is a match for the faces of known person
        encodings = [person.encoding for person in persons]
        distances = face_recognition.face_distance(encodings, face.encoding)
        index = np.argmin(distances)
        min_value = distances[index]
        if min_value < self.similarity_threshold:
            # face of known person
            persons[index].add_face(face)
            # re-calculate encoding
            persons[index].calculate_average_encoding()
            face.name = persons[index].name
            return persons[index]

    def compare_with_unknown_faces(self, face, unknown_faces):
        if len(unknown_faces) == 0:
            # this is the first face
            unknown_faces.append(face)
            face.name = "unknown"
            return

        encodings = [face.encoding for face in unknown_faces]
        distances = face_recognition.face_distance(encodings, face.encoding)
        index = np.argmin(distances)
        min_value = distances[index]
        if min_value < self.similarity_threshold:
            # two faces are similar - create new person with two faces
            person = Person()
            newly_known_face = unknown_faces.pop(index)
            person.add_face(newly_known_face)
            person.add_face(face)
            person.calculate_average_encoding()
            face.name = person.name
            newly_known_face.name = person.name
            return person
        else:
            # unknown face
            unknown_faces.append(face)
            face.name = "unknown"
            return None

    def draw_name(self, frame, face, pre_face_encoding, similarity_threshold=0.6):
        """얼굴을 기준 얼굴과 비교하고 다르면 모자이크 처리"""
        color = (0, 255, 0)  # Green for matching face
        thickness = 2
        (top, right, bottom, left) = face.location

        # 기준 얼굴과 비교
        if pre_face_encoding is not None:
            distance = face_recognition.face_distance([pre_face_encoding], face.encoding)[0]
            if distance > similarity_threshold:
                # 기준 얼굴과 다를 경우 모자이크 처리
                face_region = frame[top:bottom, left:right]
                mosaic_scale = 0.1
                small = cv2.resize(face_region, (0, 0), fx=mosaic_scale, fy=mosaic_scale)
                mosaic = cv2.resize(small, (right - left, bottom - top), interpolation=cv2.INTER_LINEAR)
                frame[top:bottom, left:right] = mosaic
                return  # 모자이크 처리 후 이름 표시 생략

        # 기준 얼굴과 동일한 경우 이름 표시
        width = 20
        if width > (right - left) // 3:
            width = (right - left) // 3
        height = 20
        if height > (bottom - top) // 3:
            height = (bottom - top) // 3
        cv2.line(frame, (left, top), (left+width, top), color, thickness)
        cv2.line(frame, (right, top), (right-width, top), color, thickness)
        cv2.line(frame, (left, bottom), (left+width, bottom), color, thickness)
        cv2.line(frame, (right, bottom), (right-width, bottom), color, thickness)
        cv2.line(frame, (left, top), (left, top+height), color, thickness)
        cv2.line(frame, (right, top), (right, top+height), color, thickness)
        cv2.line(frame, (left, bottom), (left, bottom-height), color, thickness)
        cv2.line(frame, (right, bottom), (right, bottom-height), color, thickness)

        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, "Matched Face", (left + 6, bottom + 30), font, 1.0,
                    (255, 255, 255), 1)



if __name__ == '__main__':
    import argparse
    import signal
    import time

    ap = argparse.ArgumentParser()
    ap.add_argument("inputfile", help="video file to detect or '0' to detect from web cam")
    ap.add_argument("-t", "--threshold", default=0.44, type=float,
                    help="threshold of the similarity (default=0.44)")
    ap.add_argument("-p", "--pre-face-dir", type=str, required=True,
                    help="directory containing pre_face images")
    ap.add_argument("-d", "--display", action='store_true',
                    help="display the frame in real time")
    args = ap.parse_args()

    # 기준 얼굴 로드
    pre_face_dir = args.pre_face_dir
    pdb = PersonDB()
    pdb.load_pre_face(pre_face_dir)  # 기준 얼굴 로드

    # 비디오 캡처 준비
    src_file = args.inputfile
    if src_file == "0":
        src_file = 0
    src = cv2.VideoCapture(src_file)
    if not src.isOpened():
        print("cannot open inputfile", src_file)
        exit(1)

    fc = FaceClassifier(args.threshold, 1.0)  # ratio=1.0 (no resize)
    running = True
    while running:
        ret, frame = src.read()
        if frame is None:
            break

        # 얼굴 탐지 및 처리
        faces = fc.detect_faces(frame)
        for face in faces:
            fc.draw_name(frame, face, pdb.pre_face_encoding)

        if args.display:
            cv2.imshow("Frame", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                running = False

    src.release()
    cv2.destroyAllWindows()
