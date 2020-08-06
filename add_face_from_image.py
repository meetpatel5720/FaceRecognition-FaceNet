import cv2
from constants import *
import dlib
from imutils import face_utils
from imutils.face_utils import FaceAligner
from facenet import get_image_paths, ImageClass

detector = dlib.get_frontal_face_detector()
shape_predictor = dlib.shape_predictor(SHAPE_PREDICTOR_PATH)
face_aligner = FaceAligner(shape_predictor, desiredFaceWidth=200)
img = cv2.imread(os.path.join(TEST_IMAGE_DIR, "IMG_0049.jpg"))

counter = 1
for face in detector(img):
    (x, y, w, h) = face_utils.rect_to_bb(face)
    cv2.putText(img, str(counter), (x + w, y), cv2.QT_FONT_NORMAL, 2, (0, 0, 255), 2, cv2.LINE_4)
    cv2.rectangle(img, (x, y), (x + w, y + h), (255, 255, 0), 3)

    cropped_face = img[y:y + h, x:x + w]
    final_face = cv2.resize(cropped_face, (182, 182), interpolation=cv2.INTER_LINEAR)
    NEW_IMAGE_DIR = os.path.join(IMAGE_DIR, str(counter))

    image_paths = get_image_paths(NEW_IMAGE_DIR)
    new_class = ImageClass(str(counter), image_paths)
    class_dir = os.path.join(CROPPED_IMAGE_DIR, new_class.name)
    if not os.path.exists(class_dir):
        os.makedirs(class_dir)
    print(os.path.join(class_dir, str(counter) + "_" + "1" + ".jpg"))
    local_count = 60
    while local_count > 0:
        cv2.imwrite(os.path.join(class_dir, str(counter) + "_" + str(local_count) + ".jpg"), final_face)
        local_count -= 1

    print("Done processing images of ", str(counter))
    counter += 1

# ----------------------------------------------------------------------------------
img = cv2.resize(img, (int(img.shape[1] / 5), int(img.shape[0] / 5)))
cv2.imshow("Camera", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
