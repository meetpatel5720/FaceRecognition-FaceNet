# FaceReco

This is real time face recognition system based on deep learning and nueral network and implemented using Tensorflow framework. 
This system is based on FaceNet which is described in ["FaceNet: A Unified Embedding for Face Recognition and Clustering"](http://arxiv.org/abs/1503.03832).

## Directory Structure
```
Root
├── class
│   └── classifier.pickle
├── cropped_img
│   └── Person1
│       ├── img1
│       ├── img2
│       ├── ....
├── images
│   └── Person1
│       ├── img1
│       ├── img2
│       ├── ....
├── model
│   ├── 20180402-114759.pb
│   ├── shape_predictor_68_face_landmarks.dat
├── add_face.py
├── constants.py
├── face_recognition_video.py
├── facenet.py
├── preprocess.py
├── train_model.py
```

## Environment
Anaconda Python 3.7.6

Library used
- dlib 19.19
- imutils 0.5.3
- numpy
- opencv-python 4.1.1.26
- pickle
- scikit-learn 0.21.3
- tensorflow(as tensorflow.compat.v1)

## Model
For dlib face detection download ```shape_predictor_68_face_landmarks.dat``` from [here](http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2).
For face recognition I used [20180402-114759](https://drive.google.com/open?id=1EXPBSXwTaqrSC0OhUdXNmKSh9qJUQ55-) pretrained model. This model
is trained on VGG2Face2 dataset consisting of ~3.3M faces and ~9000 classes.

## Steps
1. Add faces in ```images/``` directory. To add face run ```add_face.py``` file. It wiil automatically process newly added faces.
2. Download necessary model from link given below and put in ```model/``` directory.
3. Run ```preprocess.py``` to preprocess all faces available in ```images/```. Here I used dlib to detection of face from images. You can also use MTCNN for face detection. Note that this will needed only when you externally add face.
4. Run ```train_model.py``` to train classifier.
5. Run ```face_recognition_video.py``` for real time face recognition.

## References
- David Sandberg's facenet implementation - https://github.com/davidsandberg/facenet
- https://github.com/AISangam/Facenet-Real-time-face-recognition-using-deep-learning-Tensorflow
