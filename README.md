File Descriptions:
train_model.py: Script to train the face recognition model. It processes the dataset, extracts features from the face images, and trains a model which is saved to face_model.yml.
face_model.yml: The file where the trained face recognition model is saved.
recognize_face.py: Script to test and run the face recognition model. It captures video from the camera, detects faces, and recognizes them using the trained model.

How It Works:
Face Detection: OpenCV's Haar Cascade classifier is used to detect faces in images.
Feature Extraction: Features are extracted from the detected faces using OpenCV.
Model Training: The extracted features are used to train a face recognition model which is saved as face_model.yml.
Face Recognition: The trained model is used to recognize faces in real-time video frames captured from the webcam.

Acknowledgements:
OpenCV: An open-source computer vision and machine learning software library.
NumPy: A fundamental package for scientific computing with Python.# face-lock
