import os

import dlib
from imutils.face_utils import FaceAligner

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
IMAGE_DIR = os.path.join(BASE_DIR, "images")
CROPPED_IMAGE_DIR = os.path.join(BASE_DIR, "cropped_img")
TEST_IMAGE_DIR = os.path.join(BASE_DIR, "test")

MODEL_DIR = os.path.join(BASE_DIR, "model")
# MODEL_PATH = os.path.join(MODEL_DIR, "20170511-185253.pb")
MODEL_PATH = os.path.join(MODEL_DIR, "20180402-114759.pb")

CLASS_DIR = os.path.join(BASE_DIR, "class")
CLASSIFIER_PATH = os.path.join(CLASS_DIR, "classifier.pickle")

SHAPE_PREDICTOR_PATH = os.path.join(MODEL_DIR, "shape_predictor_68_face_landmarks.dat")
# MODEL_PATH = os.path.join(MODEL_DIR, "model.h5")
# HAARCASCADE_PATH = os.path.join(MODEL_DIR, "haarcascade_frontalface_default.xml")


detector = dlib.get_frontal_face_detector()
shape_predictor = dlib.shape_predictor(SHAPE_PREDICTOR_PATH)
face_aligner = FaceAligner(shape_predictor, desiredFaceWidth=200)
