import cv2
import mediapipe as mp
import numpy as np
import csv
import os

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils


DATA_DIR = 'sign_language_data'
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)


hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)


cap = cv2.VideoCapture(0)

labels = ["A", "B", "C", "D"] 
for label in labels:
    print(f"Collecting data for {label}...")
    data = []
    for i in range(300):  
        ret, frame = cap.read()
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(image)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                landmarks = []
                for lm in hand_landmarks.landmark:
                    landmarks.extend([lm.x, lm.y, lm.z]) 
                
                data.append(landmarks)
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        cv2.putText(frame, f"Collecting {label}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow("Data Collection", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    with open(f'{DATA_DIR}/{label}.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(data)

cap.release()
cv2.destroyAllWindows()
