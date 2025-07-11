from numba import jit, cuda
import cv2
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import neighbors
from sklearn.model_selection import train_test_split
import mediapipe as mp
import time
import pyttsx3
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands
import time
from collections import Counter
engine=pyttsx3.init()
engine. setProperty("rate", 100)

pTime = 0
cTime = 0
# For webcam input:

# Get the directory of the current script
script_dir = os.path.dirname(os.path.abspath(__file__))

# Combine the script directory with the file name
file_path = os.path.join(script_dir, "combined_csv.csv")

# Read the CSV file
df = pd.read_csv(file_path, index_col=False)

hands = mp_hands.Hands(
    min_detection_confidence=0.5, min_tracking_confidence=0.5, max_num_hands=1
)

cap = cv2.VideoCapture(0)

print("Training KNN clasifier")
X = np.array(df.drop(["Lable", "Location"], axis=1))
y = np.array(df["Lable"])
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
classifier = neighbors.KNeighborsClassifier(n_neighbors=8)
classifier.fit(X_train, y_train)
mychar=""
letter_array=['a','b','c','d','e','f','g','h','i','j']
string_array=" "
final=''
char_val=" "
count_int=0
running = True
while running:
    success, image = cap.read()
    # #if not success:
    #     print("Ignoring empty camera frame.")
    #     # If loading a video, use 'break' instead of 'continue'.
    #     break

    # Flip the image horizontally for a later selfie-view display, and convert
    # the BGR image to RGB.
    image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
    # To improve performance, optionally mark the image as not writeable to
    # pass by reference.
    image.flags.writeable = False
    results = hands.process(image)

    # Draw the hand annotations on the image.
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            x0 = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].x
            y0 = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].y
            x1 = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_CMC].x
            y1 = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_CMC].y
            x2 = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_MCP].x
            y2 = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_MCP].y
            x3 = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_IP].x
            y3 = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_IP].y
            x4 = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].x
            y4 = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].y
            x5 = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP].x
            y5 = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP].y
            x6 = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_PIP].x
            y6 = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_PIP].y
            x7 = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_DIP].x
            y7 = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_DIP].y
            x8 = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x
            y8 = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y
            x9 = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_MCP].x
            y9 = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_MCP].y
            x10 = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_PIP].x
            y10 = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_PIP].y
            x11 = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_DIP].x
            y11 = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_DIP].y
            x12 = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].x
            y12 = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].y
            x13 = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_MCP].x
            y13 = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_MCP].y
            x14 = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_PIP].x
            y14 = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_PIP].y
            x15 = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_DIP].x
            y15 = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_DIP].y
            x16 = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP].x
            y16 = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP].y
            x17 = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_MCP].x
            y17 = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_MCP].y
            x18 = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_PIP].x
            y18 = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_PIP].y
            x19 = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_DIP].x
            y19 = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_DIP].y
            x20 = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP].x
            y20 = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP].y
            o1 = np.linalg.norm(np.array((x1, y1)) - np.array((x0, y0)))
            o2 = np.linalg.norm(np.array((x2, y2)) - np.array((x0, y0)))
            o3 = np.linalg.norm(np.array((x3, y3)) - np.array((x0, y0)))
            o4 = np.linalg.norm(np.array((x4, y4)) - np.array((x0, y0)))
            o5 = np.linalg.norm(np.array((x5, y5)) - np.array((x0, y0)))
            o6 = np.linalg.norm(np.array((x6, y6)) - np.array((x0, y0)))
            o7 = np.linalg.norm(np.array((x7, y7)) - np.array((x0, y0)))
            o8 = np.linalg.norm(np.array((x8, y8)) - np.array((x0, y0)))
            o9 = np.linalg.norm(np.array((x9, y9)) - np.array((x0, y0)))
            o10 = np.linalg.norm(np.array((x10, y10)) - np.array((x0, y0)))
            o11 = np.linalg.norm(np.array((x11, y11)) - np.array((x0, y0)))
            o12 = np.linalg.norm(np.array((x12, y12)) - np.array((x0, y0)))
            o13 = np.linalg.norm(np.array((x13, y13)) - np.array((x0, y0)))
            o14 = np.linalg.norm(np.array((x14, y14)) - np.array((x0, y0)))
            o15 = np.linalg.norm(np.array((x15, y15)) - np.array((x0, y0)))
            o16 = np.linalg.norm(np.array((x16, y16)) - np.array((x0, y0)))
            o17 = np.linalg.norm(np.array((x17, y17)) - np.array((x0, y0)))
            o18 = np.linalg.norm(np.array((x18, y18)) - np.array((x0, y0)))
            o19 = np.linalg.norm(np.array((x19, y19)) - np.array((x0, y0)))
            o20 = np.linalg.norm(np.array((x20, y20)) - np.array((x0, y0)))

            handd = [
                [
                    x0,
                    y0,
                    x1,
                    y1,
                    x2,
                    y2,
                    x3,
                    y3,
                    x4,
                    y4,
                    x5,
                    y5,
                    x6,
                    y6,
                    x7,
                    y7,
                    x8,
                    y8,
                    x9,
                    y9,
                    x10,
                    y10,
                    x11,
                    y11,
                    x12,
                    y12,
                    x13,
                    y13,
                    x14,
                    y14,
                    x15,
                    y15,
                    x16,
                    y16,
                    x17,
                    y17,
                    x18,
                    y18,
                    x19,
                    y19,
                    x20,
                    y20,
                    o1,
                    o2,
                    o3,
                    o4,
                    o5,
                    o6,
                    o7,
                    o8,
                    o9,
                    o10,
                    o11,
                    o12,
                    o13,
                    o14,
                    o15,
                    o16,
                    o17,
                    o18,
                    o19,
                    o20,
                ]
            ]

            df2 = pd.DataFrame(
                handd,
                columns=[
                    "x0",
                    "y0",
                    "x1",
                    "y1",
                    "x2",
                    "y2",
                    "x3",
                    "y3",
                    "x4",
                    "y4",
                    "x5",
                    "y5",
                    "x6",
                    "y6",
                    "x7",
                    "y7",
                    "x8",
                    "y8",
                    "x9",
                    "y9",
                    "x10",
                    "y10",
                    "x11",
                    "y11",
                    "x12",
                    "y12",
                    "x13",
                    "y13",
                    "x14",
                    "y14",
                    "x15",
                    "y15",
                    "x16",
                    "y16",
                    "x17",
                    "y17",
                    "x18",
                    "y18",
                    "x19",
                    "y19",
                    "x20",
                    "y20",
                    "0_1",
                    "0_2",
                    "0_3",
                    "0_4",
                    "0_5",
                    "0_6",
                    "0_7",
                    "0_8",
                    "0_9",
                    "0_10",
                    "0_11",
                    "0_12",
                    "0_13",
                    "0_14",
                    "0_15",
                    "0_16",
                    "0_17",
                    "0_18",
                    "0_19",
                    "0_20",
                ],
            )

            letter = classifier.predict(np.array(df2))
            char_val=str(letter[0])
            print(f"\r\nLetter: {letter[0]}", end="")
            letter_array[count_int]=str(char_val)
            count_int=count_int+1
            if count_int>9:
                    print("\r\nChar Array:")
                    print(letter_array)
                    d = Counter(letter_array)
                    print(d)
                    new_list = ([item for item in d if d[item]>5])
                    mychar=(str(new_list)[1:-1])
                    mychar=mychar.strip(" ")
                    print(mychar)
                    count_int=0
                    
                    #print(char_val)
                    if "Clear all" in mychar:
                        string_array=""
                    elif "L" in mychar:
                        final=string_array
                    elif "Space" in mychar:
                        string_array+=" "
                    elif "Erase" in mychar:
                        string_array=string_array[:-1]
                        print("\r\nRecent Char has been deleted")
                    else:
                        string_array+=str(mychar.strip("'"))
                  
    # fps = cvFpsCalc.get()
    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime
    
    cv2.putText(
        image,
        "Char:" + (str(char_val.strip("'"))),
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (0, 0, 0),
        4,
        cv2.LINE_AA,
    )
    cv2.putText(
        image,
        "Char:" + (str(char_val.strip("'"))),
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (0, 0, 0),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        image,
        "String: " + str(string_array.strip(None)),
        (10, 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (0, 0, 0),
        4,
        cv2.LINE_AA,
    )
    cv2.putText(
        image,
        "String: " + str(string_array.strip(None)),
        (10, 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (0, 0, 0),#<--changed from 255,255,255 to 000
        2,
        cv2.LINE_AA,
    )
    cv2.imshow("Sign Language Recognition ", image)
    engine.say(final)
    engine.runAndWait()
    time.sleep(0)
    # if cv2.waitKey(5) & 0xFF == 1:
    if cv2.waitKey(1) == 27:
        running = False

hands.close()
cap.release()
