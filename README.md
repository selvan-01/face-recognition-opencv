# Face Recognition using OpenCV

This project implements a real-time face recognition system using Python and OpenCV.  
It can detect and recognize faces from images, videos, or live webcam streams.

## Features

- Real-time face detection using Haar Cascade
- Face dataset creation using webcam
- Face recognition using FisherFaceRecognizer
- Detects known and unknown persons
- Automatically captures unknown faces

## Technologies Used

- Python
- OpenCV
- NumPy

## Project Structure

face-recognition-opencv
│
├── datasets
│ ├── Elon
│ └── Steve
│
├── create_data.py
├── face_recognize.py
├── haarcascade_frontalface_default.xml
├── requirements.txt
└── README.md


## Installation

Clone the repository: https://github.com/selvan-01/face-recognition-opencv.git


Install dependencies:

pip install -r requirements.txt


## Step 1: Collect Dataset

Run the dataset creation script:

python create_data.py

This will capture 30 face images and store them in the dataset folder.

## Step 2: Run Face Recognition

python face_recognize.py


The system will detect and recognize faces in real-time using your webcam.

## Output

- Displays recognized person's name
- Shows "Unknown" for unidentified faces
- Captures image of unknown person

## Author
S. Senthamil Selvan  
AI Developer | Computer Science Engineer