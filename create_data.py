"""
create_data.py
----------------
This script captures face images using a webcam and
stores them in the dataset folder for training.

Author: S. Senthamil Selvan
"""

import cv2
import os

# -------------------------------
# Configuration
# -------------------------------

haar_file = 'haarcascade_frontalface_default.xml'
dataset_path = 'datasets'
person_name = 'Elon'   # Change this name for different people
image_width, image_height = (130, 100)
max_images = 30

# -------------------------------
# Create Dataset Folder
# -------------------------------

path = os.path.join(dataset_path, person_name)

if not os.path.isdir(path):
    os.makedirs(path)

# -------------------------------
# Load Face Detection Model
# -------------------------------

face_cascade = cv2.CascadeClassifier(haar_file)

# Start Webcam
webcam = cv2.VideoCapture(0)

count = 1

print("Starting face data collection...")

# -------------------------------
# Capture Images
# -------------------------------

while count <= max_images:

    ret, frame = webcam.read()
    if not ret:
        break

    # Convert frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = face_cascade.detectMultiScale(gray, 1.3, 4)

    for (x, y, w, h) in faces:

        # Draw rectangle around face
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

        # Extract face region
        face = gray[y:y+h, x:x+w]

        # Resize face
        face_resize = cv2.resize(face, (image_width, image_height))

        # Save image
        cv2.imwrite(f"{path}/{count}.png", face_resize)

        count += 1

    # Show camera feed
    cv2.imshow("Face Data Collection", frame)

    # Press ESC to exit
    key = cv2.waitKey(10)
    if key == 27:
        break

# -------------------------------
# Cleanup
# -------------------------------

print("Dataset collection completed.")

webcam.release()
cv2.destroyAllWindows()