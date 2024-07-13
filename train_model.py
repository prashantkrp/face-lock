import cv2
import numpy as np
import os

# Define the directory containing face images
face_dir = 'faces'

# Check if the directory exists and contains images
if not os.path.exists(face_dir):
    raise FileNotFoundError(f"The directory {face_dir} does not exist.")
if not os.listdir(face_dir):
    raise FileNotFoundError(f"No images found in the directory {face_dir}.")

face_images = []
face_labels = []
label = 0

# Load images from the directory
for filename in os.listdir(face_dir):
    if filename.endswith('.jpg'):
        img_path = os.path.join(face_dir, filename)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is not None:
            face_images.append(img)
            face_labels.append(label)
        else:
            print(f"Warning: Unable to read image {img_path}")

# Check if images were loaded
if not face_images:
    raise ValueError("No valid face images were loaded for training.")

# Train the face recognizer
try:
    face_recognizer = cv2.face.LBPHFaceRecognizer_create()
    face_recognizer.train(face_images, np.array(face_labels))
    # Save the trained model
    model_path = 'face_model.yml'
    face_recognizer.save(model_path)
    print(f"Model trained and saved at {model_path}")
except cv2.error as e:
    print(f"Error in training the face recognizer: {e}")
