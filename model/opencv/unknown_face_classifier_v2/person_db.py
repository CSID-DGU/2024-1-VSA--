#!/usr/bin/env python3

import os
import cv2
import imutils
import shutil
import face_recognition
import numpy as np
import time
import pickle


class Face():
    key = "face_encoding"

    def __init__(self, filename, image, face_encoding):
        self.filename = filename
        self.image = image
        self.encoding = face_encoding

    def save(self, base_dir):
        # save image
        pathname = os.path.join(base_dir, self.filename)
        cv2.imwrite(pathname, self.image)

    @classmethod
    def get_encoding(cls, image):
        rgb = image[:, :, ::-1]
        boxes = face_recognition.face_locations(rgb, model="hog")
        if not boxes:
            height, width, channels = image.shape
            top = int(height/3)
            bottom = int(top*2)
            left = int(width/3)
            right = int(left*2)
            box = (top, right, bottom, left)
        else:
            box = boxes[0]
        return face_recognition.face_encodings(image, [box])[0]


class Person():
    _last_id = 0

    def __init__(self, name=None):
        if name is None:
            Person._last_id += 1
            self.name = "person_%02d" % Person._last_id
        else:
            self.name = name
            if name.startswith("person_") and name[7:].isdigit():
                id = int(name[7:])
                if id > Person._last_id:
                    Person._last_id = id
        self.encoding = None
        self.faces = []

    def add_face(self, face):
        # add face
        self.faces.append(face)

    def calculate_average_encoding(self):
        if len(self.faces) is 0:
            self.encoding = None
        else:
            encodings = [face.encoding for face in self.faces]
            self.encoding = np.average(encodings, axis=0)

    def distance_statistics(self):
        encodings = [face.encoding for face in self.faces]
        distances = face_recognition.face_distance(encodings, self.encoding)
        return min(distances), np.mean(distances), max(distances)

    def save_faces(self, base_dir):
        pathname = os.path.join(base_dir, self.name)
        try:
            shutil.rmtree(pathname)
        except OSError as e:
            pass
        os.mkdir(pathname)
        for face in self.faces:
            face.save(pathname)

    def save_montages(self, base_dir):
        images = [face.image for face in self.faces]
        montages = imutils.build_montages(images, (128, 128), (6, 2))
        for i, montage in enumerate(montages):
            filename = "montage." + self.name + ("-%02d.png" % i)
            pathname = os.path.join(base_dir, filename)
            cv2.imwrite(pathname, montage)

    @classmethod
    def load(cls, pathname, face_encodings):
        basename = os.path.basename(pathname)
        person = Person(basename)
        for face_filename in os.listdir(pathname):
            face_pathname = os.path.join(pathname, face_filename)
            image = cv2.imread(face_pathname)
            if image.size == 0:
                continue
            if face_filename in face_encodings:
                face_encoding = face_encodings[face_filename]
            else:
                print(pathname, face_filename, "calculate encoding")
                face_encoding = Face.get_encoding(image)
            if face_encoding is None:
                print(pathname, face_filename, "drop face")
            else:
                face = Face(face_filename, image, face_encoding)
                person.faces.append(face)
        print(person.name, "has", len(person.faces), "faces")
        person.calculate_average_encoding()
        return person

class PersonDB():
    def __init__(self):
        self.persons = []
        self.unknown_dir = "unknowns"
        self.encoding_file = "face_encodings"
        self.unknown = Person(self.unknown_dir)
        self.pre_face_encoding = None  # 기준 얼굴 인코딩 저장

    def load_pre_face(self, pre_face_dir):
        """pre_face 디렉토리에서 얼굴 인코딩 로드"""
        if not os.path.isdir(pre_face_dir):
            print(f"{pre_face_dir} 디렉토리가 존재하지 않습니다.")
            return
        
        for entry in os.listdir(pre_face_dir):
            pre_face_path = os.path.join(pre_face_dir, entry)
            image = cv2.imread(pre_face_path)
            if image is None or image.size == 0:
                continue
            
            encoding = Face.get_encoding(image)
            if encoding is not None:
                self.pre_face_encoding = encoding
                print(f"기준 얼굴 로드 완료: {entry}")
                return
        
        print(f"{pre_face_dir}에 유효한 얼굴 사진이 없습니다.")



if __name__ == '__main__':
    dir_name = "result"
    pdb = PersonDB()
    pdb.load_db(dir_name)
    pdb.print_persons()
    pdb.save_montages(dir_name)
    pdb.save_encodings(dir_name)