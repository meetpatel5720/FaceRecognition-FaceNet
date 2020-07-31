import os

import cv2

from constants import *
from random import randint

video = cv2.VideoCapture(1)

while True:
    check, frame = video.read()

    cv2.imshow("Camera", frame)
    key = cv2.waitKey(int(1000 / 60))
    if key == ord('q'):
        break
    elif key == ord('s'):
        rand_num = randint(10000, 99999)
        print(os.path.join(TEST_IMAGE_DIR, str(rand_num) + ".jpg"))
        cv2.imwrite(os.path.join(TEST_IMAGE_DIR, str(rand_num) + ".jpg"), frame)
        break

video.release()
cv2.destroyAllWindows()
