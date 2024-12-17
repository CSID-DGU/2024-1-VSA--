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
            top = int(height / 3)
            bottom = int(top * 2)
            left = int(width / 3)
            right = int(left * 2)
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
        if len(self.faces) == 0:
            self.encoding = None
        else:
            encodings = [face.encoding for face in self.faces]
            self.encoding = np.average(encodings, axis=0)

    def save_faces(self, base_dir):
        pathname = os.path.join(base_dir, self.name)
        try:
            shutil.rmtree(pathname)
        except OSError:
            pass
        os.mkdir(pathname)
        for face in self.faces:
            face.save(pathname)


class PersonDB():
    def __init__(self):
        self.persons = []
        self.first_person = None

    def load_db(self, dir_name):
        if not os.path.isdir(dir_name):
            return
        print("Start loading persons in the directory '%s'" % dir_name)
        start_time = time.time()

        # Read persons
        for entry in os.scandir(dir_name):
            if entry.is_dir(follow_symlinks=False):
                pathname = os.path.join(dir_name, entry.name)
                person = Person.load(pathname, {})
                if len(person.faces) == 0:
                    continue
                if self.first_person is None:
                    self.first_person = person
                else:
                    self.persons.append(person)
        elapsed_time = time.time() - start_time
        print("Loading persons finished in %.3f sec." % elapsed_time)

    def save_db(self, dir_name):
        print("Start saving persons in the directory '%s'" % dir_name)
        start_time = time.time()
        try:
            shutil.rmtree(dir_name)
        except OSError:
            pass
        os.mkdir(dir_name)

        if self.first_person:
            self.first_person.save_faces(dir_name)

        elapsed_time = time.time() - start_time
        print("Saving persons finished in %.3f sec." % elapsed_time)

    def is_first_person(self, face):
        if self.first_person is None:
            return False
        distances = face_recognition.face_distance([self.first_person.encoding], face.encoding)
        return distances[0] < 0.6  # Adjust threshold as needed

    def print_persons(self):
        if self.first_person:
            print(f"First person: {self.first_person.name} with {len(self.first_person.faces)} faces")


if __name__ == '__main__':
    dir_name = "result"
    pdb = PersonDB()
    pdb.load_db(dir_name)
    pdb.print_persons()
    pdb.save_db(dir_name)
