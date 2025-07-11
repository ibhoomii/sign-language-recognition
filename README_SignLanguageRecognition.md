
# 🤟 Sign Language Recognition using Mediapipe & KNN

A real-time vision-based Sign Language Recognition system that converts American Sign Language (ASL) hand gestures into text and speech. Built using Python, MediaPipe, OpenCV, and K-Nearest Neighbors (KNN) algorithm.

---

## 🧠 Project Abstract

This project aims to bridge the communication gap between speech/hearing-impaired individuals and the general public. It recognizes static hand gestures representing A–Z alphabets in ASL and converts them into text on screen and speech using a Text-to-Speech engine.

---

## ⚙️ Features

- Real-time hand gesture recognition using webcam
- ASL A-Z alphabet detection
- Word formation using continuous gestures
- Text-to-Speech output of recognized words
- Erase functionality to correct typed words

---

## 🛠️ Tech Stack

- Python
- MediaPipe
- OpenCV
- NumPy
- Pandas
- KNN Algorithm
- pyttsx3 (TTS)
- CSV for dataset handling

---


## 🚀 How It Works

1. **Image Acquisition**: Captures hand gestures using webcam.
2. **Landmark Detection**: Uses MediaPipe to extract 21 hand landmarks (x, y).
3. **Feature Engineering**: Calculates 20 Euclidean distances from wrist to each point.
4. **Classification**: Applies KNN algorithm with K=8 to classify the gesture.
5. **Output Display**:
    - Displays detected alphabet on-screen (`Char`)
    - Forms a word (`String`)
    - Speaks out word using TTS
6. **Erase Gesture**: Erases last character if Erase gesture is detected.


---

## 📈 Dataset Preparation

- 40 images per alphabet (A–Z) captured using webcam.
- Each image processed to extract 21 landmark points using MediaPipe.
- Saved to `.csv` using `landmarkextract.py`.
- Combined into `datasets.csv` and labeled accordingly.

---

## 📊 Algorithm Used

**K-Nearest Neighbors (KNN)**
- K = 8
- Classifies gesture based on nearest 8 vectors
- Chosen for its simplicity and real-time performance

---

## 🔊 Additional Functionalities

- **Text-to-Speech** using `pyttsx3`
- **Erase Gesture** to remove last character
- **String Formation** for words like “HOW ARE YOU”

---