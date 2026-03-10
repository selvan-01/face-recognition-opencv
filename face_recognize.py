"""
face_recognize.py
-----------------
This script trains a face recognition model using
the collected dataset and performs real-time
face recognition using a webcam.

Author: S. Senthamil Selvan
"""

import cv2
import numpy as np
import os

# -------------------------------
# Configuration
# -------------------------------

haar_file = 'haarcascade_frontalface_default.xml'
dataset_path = 'datasets'
image_width, image_height = (130, 100)

print("Training the face recognition model...")

# -------------------------------
# Prepare Dataset
# -------------------------------

images = []
labels = []
names = {}
current_id = 0

# Load dataset images
for subdirs, dirs, files in os.walk(dataset_path):

    for subdir in dirs:

        names[current_id] = subdir
        subject_path = os.path.join(dataset_path, subdir)

        for filename in os.listdir(subject_path):

            image_path = os.path.join(subject_path, filename)

            img = cv2.imread(image_path, 0)

            images.append(img)
            labels.append(current_id)

        current_id += 1

# Convert lists to numpy arrays
images = np.array(images)
labels = np.array(labels)

# -------------------------------
# Train Face Recognition Model
# -------------------------------

model = cv2.face.FisherFaceRecognizer_create()
model.train(images, labels)

print("Training completed.")

# -------------------------------
# Start Face Recognition
# -------------------------------

face_cascade = cv2.CascadeClassifier(haar_file)
webcam = cv2.VideoCapture(0)

unknown_counter = 0

print("Starting real-time face recognition...")

while True:

    ret, frame = webcam.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:

        face = gray[y:y+h, x:x+w]
        face_resize = cv2.resize(face, (image_width, image_height))

        prediction = model.predict(face_resize)

        confidence = prediction[1]

        # Draw rectangle
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 3)

        # Recognized face
        if confidence < 800:

            name = names[prediction[0]]

            cv2.putText(
                frame,
                f"{name} - {confidence:.0f}",
                (x-10, y-10),
                cv2.FONT_HERSHEY_COMPLEX,
                1,
                (51, 255, 255)
            )

            print(name)
            unknown_counter = 0

        # Unknown face
        else:

            unknown_counter += 1

            cv2.putText(
                frame,
                "Unknown",
                (x-10, y-10),
                cv2.FONT_HERSHEY_PLAIN,
                1,
                (0, 255, 0)
            )

            if unknown_counter > 100:

                print("Unknown Person Detected")
                cv2.imwrite("input.jpg", frame)
                unknown_counter = 0

    cv2.imshow("Face Recognition", frame)

    key = cv2.waitKey(10)

    if key == 27:
        break

# -------------------------------
# Cleanup
# -------------------------------

webcam.release()
cv2.destroyAllWindows()