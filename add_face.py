import time
import cv2
import os

from imutils import face_utils

from constants import *
from facenet import get_image_paths, ImageClass

name = input("Enter name of person - ")

NEW_IMAGE_DIR = os.path.join(IMAGE_DIR, name)
if not os.path.exists(NEW_IMAGE_DIR):
    os.makedirs(NEW_IMAGE_DIR)
print(NEW_IMAGE_DIR)
video = cv2.VideoCapture(0)
i = 0
while i < 30:
    check, frame = video.read()

    cv2.imshow("Camera", frame)
    time.sleep(0.2)
    key = cv2.waitKey(int(1000 / 60))
    print(os.path.join(NEW_IMAGE_DIR, name + "_" + str(i) + ".jpg"))
    cv2.imwrite(os.path.join(NEW_IMAGE_DIR, name + "_" + str(i) + ".jpg"), frame)
    i += 1
    if key == ord('q'):
        break

video.release()
cv2.destroyAllWindows()

print("Processing images of ", name)
image_paths = get_image_paths(NEW_IMAGE_DIR)
new_class = ImageClass(name, image_paths)
for img_path in new_class.image_paths:
    img = cv2.imread(img_path)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    face = detector(img, 1)
    if len(face) > 0:
        face = detector(img, 1)[0]
        (x, y, w, h) = face_utils.rect_to_bb(face)
        face_img = img[y:y + h, x:x + w]

        final_face = cv2.resize(face_img, (182, 182), interpolation=cv2.INTER_LINEAR)
        class_dir = os.path.join(CROPPED_IMAGE_DIR, new_class.name)
        if not os.path.exists(class_dir):
            os.makedirs(class_dir)
        print(os.path.join(class_dir, os.path.basename(img_path)))
        cv2.imwrite(os.path.join(class_dir, os.path.basename(img_path)), final_face)
print("Done processing images of ", name)

