#!/usr/bin/env python3
# python face_classifier.py 0 --display
# 객체당 얼굴 임베딩 개수 20개로 제한  프레임 인터벌 5
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
        if self.ratio == 1.0:
            rgb = frame[:, :, ::-1]
        else:
            small_frame = cv2.resize(frame, (0, 0), fx=self.ratio, fy=self.ratio)
            rgb = small_frame[:, :, ::-1]
        boxes = face_recognition.face_locations(rgb)
        if self.ratio == 1.0:
            return boxes
        boxes_org_size = []
        for box in boxes:
            (top, right, bottom, left) = box
            left = int(left / self.ratio)
            right = int(right / self.ratio)
            top = int(top / self.ratio)
            bottom = int(bottom / self.ratio)
            box_org_size = (top, right, bottom, left)
            boxes_org_size.append(box_org_size)
        return boxes_org_size

    def detect_faces(self, frame):
        boxes = self.locate_faces(frame)
        if len(boxes) == 0:
            return []

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

        encodings = [person.encoding for person in persons]
        distances = face_recognition.face_distance(encodings, face.encoding)
        index = np.argmin(distances)
        min_value = distances[index]
        if min_value < self.similarity_threshold:
            persons[index].add_face(face)
            persons[index].calculate_average_encoding()
            face.name = persons[index].name
            return persons[index]

    def compare_with_unknown_faces(self, face, unknown_faces):
        if len(unknown_faces) == 0:
            unknown_faces.append(face)
            face.name = "unknown"
            return

        encodings = [face.encoding for face in unknown_faces]
        distances = face_recognition.face_distance(encodings, face.encoding)
        index = np.argmin(distances)
        min_value = distances[index]
        if min_value < self.similarity_threshold:
            person = Person()
            newly_known_face = unknown_faces.pop(index)
            person.add_face(newly_known_face)
            person.add_face(face)
            person.calculate_average_encoding()
            face.name = person.name
            newly_known_face.name = person.name
            return person
        else:
            unknown_faces.append(face)
            face.name = "unknown"
            return None

    def draw_name(self, frame, face):
        color = (0, 0, 255)
        thickness = 2
        (top, right, bottom, left) = face.location

        # draw box
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

        # If the face is not person_01, apply mosaic
        if face.name != "person_01":
            face_region = frame[top:bottom, left:right]
            mosaic_scale = 0.1
            small = cv2.resize(face_region, (0, 0), fx=mosaic_scale, fy=mosaic_scale)
            mosaic = cv2.resize(small, (right - left, bottom - top), interpolation=cv2.INTER_LINEAR)
            frame[top:bottom, left:right] = mosaic
        else:
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(frame, face.name, (left + 6, bottom + 30), font, 1.0,
                        (255, 255, 255), 1)


if __name__ == '__main__':
    import argparse
    import signal
    import time
    import os

    ap = argparse.ArgumentParser()
    ap.add_argument("inputfile",
                    help="video file to detect or '0' to detect from web cam")
    ap.add_argument("-t", "--threshold", default=0.44, type=float,
                    help="threshold of the similarity (default=0.44)")
    ap.add_argument("-d", "--display", action='store_true',
                    help="display the frame in real time")
    ap.add_argument("-r", "--resize-ratio", default=1.0, type=float,
                    help="resize the frame to process (less time, less accuracy)")
    args = ap.parse_args()

    src_file = args.inputfile
    if src_file == "0":
        src_file = 0

    src = cv2.VideoCapture(src_file)
    if not src.isOpened():
        print("cannot open inputfile", src_file)
        exit(1)

    frame_width = src.get(cv2.CAP_PROP_FRAME_WIDTH)
    frame_height = src.get(cv2.CAP_PROP_FRAME_HEIGHT)
    frame_rate = src.get(5)

    # Set detection frame interval to 5
    frames_between_capture = 5

    print("source", args.inputfile)
    print("original: %dx%d, %f frame/sec" % (src.get(3), src.get(4), frame_rate))
    ratio = float(args.resize_ratio)
    if ratio != 1.0:
        s = "RESIZE_RATIO: " + args.resize_ratio
        s += " -> %dx%d" % (int(src.get(3) * ratio), int(src.get(4) * ratio))
        print(s)
    print("process every %d frame" % frames_between_capture)
    print("similarity threshold:", args.threshold)

    result_dir = "result"
    pdb = PersonDB()
    pdb.load_db(result_dir)
    pdb.print_persons()

    fc = FaceClassifier(args.threshold, ratio)
    frame_id = 0
    running = True

    while running:
        ret, frame = src.read()
        if frame is None:
            break

        frame_id += 1
        if frame_id % frames_between_capture != 0:
            continue

        faces = fc.detect_faces(frame)
        for face in faces:
            person = fc.compare_with_known_persons(face, pdb.persons)
            if person:
                continue
            person = fc.compare_with_unknown_faces(face, pdb.unknown.faces)
            if person:
                pdb.persons.append(person)

        if args.display:
            for face in faces:
                fc.draw_name(frame, face)
            cv2.imshow("Frame", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    src.release()
    cv2.destroyAllWindows()
