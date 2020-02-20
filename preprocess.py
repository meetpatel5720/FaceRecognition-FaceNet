import cv2
from imutils import face_utils
from constants import *
import facenet

if not os.path.exists(CROPPED_IMAGE_DIR):
    os.makedirs('cropped_img')

dataset = facenet.get_dataset(IMAGE_DIR)

for img_cls in dataset:
    for img_path in img_cls.image_paths:
        img = cv2.imread(img_path)
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        face = detector(img, 1)
        if len(face) > 0:
            face = detector(img, 1)[0]
            (x, y, w, h) = face_utils.rect_to_bb(face)
            face_img = img[y:y + h, x:x + w, :]

            final_face = cv2.resize(face_img, (182, 182), interpolation=cv2.INTER_LINEAR)
            class_dir = os.path.join(CROPPED_IMAGE_DIR, img_cls.name)
            if not os.path.exists(class_dir):
                os.makedirs(class_dir)
            print(os.path.join(class_dir, os.path.basename(img_path)))
            cv2.imwrite(os.path.join(class_dir, os.path.basename(img_path)), final_face)
