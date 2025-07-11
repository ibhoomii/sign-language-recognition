#After step hand_pattern_capture step follow below steps rto convert the same into digital point form
#1) save this script in pre-created folder named "DataSet_Pics"
#2) Run this script with read path = "loadImages(path = "/A/"):  # read path" and also change letter for different letter as mentioned in this code commented
#3) data.txt will be generated after running this script. Copy that file to another location and rename as dataA.txt for letter "A" and so on
#4) Repeat the same steps for all letters.

import cv2
import os
import sys
import pathlib
import numpy as np
import pandas as pd
import mediapipe as mp
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

print('x0,y0,x1,y1,x2,y2,x3,y3,x4,y4,x5,y5,x6,y6,x7,y7,x8,y8,x9,y9,x10,y10,x11,y11,x12,y12,x13,y13,x14,y14,x15,y15,x16,y16,x17,y17,x18,y18,x19,y19,x20,y20,Lable,Location')


def loadImages(path = "C:\Users\Bhoomi\Desktop\ALL FINAL\DataSet_Pics\Z"):  # read path 1
    return [os.path.join(path, f) for f in os.listdir(path) if f.endswith('.jpg')]

file_list = loadImages()
# For static images:
hands = mp_hands.Hands(
    static_image_mode=True,
    max_num_hands=1,
    min_detection_confidence=0.2)



for idx, file in enumerate(file_list):
  # Read an image, flip it around y-axis for correct handedness output (see
  # aCove).
  
  image = cv2.flip(cv2.imread(file), 1)
  # Convert the BGR image to RGB before processing.
  results = hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

  # Print handedness and draw hand landmarks on the image.
  #print('Handedness:', results.multi_handedness)
  if not results.multi_hand_landmarks:
    continue
  image_hight, image_width, _ = image.shape
  annotated_image = image.copy()

  for hand_landmarks in results.multi_hand_landmarks:
    #print(hand_landmarks)
    #x = [landmark for landmark in hand_landmarks.landmark]
    #x = [landmark for landmark in hand_landmarks.landmark]

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
    #euclidean distance
    #Let Consider Writst is the Origin (x,y)having points so from here lets find distance
    o1 = np.linalg.norm(np.array((x1,y1)) - np.array((x0,y0)))
    o2 = np.linalg.norm(np.array((x2,y2)) - np.array((x0,y0)))
    o3 = np.linalg.norm(np.array((x3,y3)) - np.array((x0,y0)))
    o4 = np.linalg.norm(np.array((x4,y4)) - np.array((x0,y0)))
    o5 = np.linalg.norm(np.array((x5,y5)) - np.array((x0,y0)))
    o6 = np.linalg.norm(np.array((x6,y6)) - np.array((x0,y0)))
    o7 = np.linalg.norm(np.array((x7,y7)) - np.array((x0,y0)))
    o8 = np.linalg.norm(np.array((x8,y8)) - np.array((x0,y0)))
    o9 = np.linalg.norm(np.array((x9,y9)) - np.array((x0,y0)))
    o10 = np.linalg.norm(np.array((x10,y10)) - np.array((x0,y0)))
    o11 = np.linalg.norm(np.array((x11,y11)) - np.array((x0,y0)))
    o12 = np.linalg.norm(np.array((x12,y12)) - np.array((x0,y0)))
    o13 = np.linalg.norm(np.array((x13,y13)) - np.array((x0,y0)))
    o14 = np.linalg.norm(np.array((x14,y14)) - np.array((x0,y0)))
    o15 = np.linalg.norm(np.array((x15,y15)) - np.array((x0,y0)))
    o16 = np.linalg.norm(np.array((x16,y16)) - np.array((x0,y0)))
    o17 = np.linalg.norm(np.array((x17,y17)) - np.array((x0,y0)))
    o18 = np.linalg.norm(np.array((x18,y18)) - np.array((x0,y0)))
    o19 = np.linalg.norm(np.array((x19,y19)) - np.array((x0,y0)))
    o20 = np.linalg.norm(np.array((x20,y20)) - np.array((x0,y0)))
    path = "C:\\Users\Rahul\\Desktop\\project sign lang\\DataSet_Pics\\Z"
    lable  ='A' # change this lable as per the folder letter <<<<<<<<< change this for every letter
    #y = [landmark.y for landmark in hand_landmarks.landmark]
    print(x0,',',y0,',',
          x1,',',y1,',',
          x2,',',y2,',',
          x3,',',y3,',',
          x4,',',y4,',',
          x5,',',y5,',',
          x6,',',y6,',',
          x7,',',y7,',',
          x8,',',y8,',',
          x9,',',y9,',',
          x10,',',y10,',',
          x11,',',y11,',',
          x12,',',y12,',',
          x13,',',y13,',',
          x14,',',y14,',',
          x15,',',y15,',',
          x16,',',y16,',',
          x17,',',y17,',',
          x18,',',y18,',',
          x19,',',y19,',',
          x20,',',y20,',',
          o1,',',
          o2,',',
          o3,',',
          o4,',',
          o5,',',
          o6,',',
          o7,',',
          o8,',',
          o9,',',
          o10,',',
          o11,',',
          o12,',',
          o13,',',
          o14,',',
          o15,',',
          o16,',',
          o17,',',
          o18,',',
          o19,',',
          o20,',' + lable,',' + path + str(idx) + '.jpg')

    #print(f'Index finger tip coordinates: (',f'{hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x}, '
      #  f'{hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y})'
    #)

    mp_drawing.draw_landmarks(annotated_image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
    cv2.imshow('hand', annotated_image)
  cv2.imwrite( os.path.join(path , 'frame' + str(idx) + '.png'), cv2.flip(annotated_image, 1))
  sys.stdout = open("dataZ.txt", "a")
  
        
hands.close()

