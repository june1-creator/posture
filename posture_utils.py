import numpy as np
import pandas as pd
import cv2
import mediapipe as mp
from datetime import datetime
import os

mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_drawing = mp.solutions.drawing_utils

def calculate_angle(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    return 360 - angle if angle > 180 else angle

def analyze_posture(landmarks):
    # Extract key points
    shoulder = [landmarks[11].x, landmarks[11].y]
    neck = [landmarks[0].x, landmarks[0].y]  # nose approx.
    ear = [landmarks[7].x, landmarks[7].y]
    left_shoulder = [landmarks[11].x, landmarks[11].y]
    right_shoulder = [landmarks[12].x, landmarks[12].y]

    # Compute angles
    neck_angle = calculate_angle(shoulder, neck, ear)
    shoulder_tilt = abs(left_shoulder[1] - right_shoulder[1]) * 1000  # scaled difference

    feedback = []

    # Classify posture
    if 95 < neck_angle < 110:
        status = "Good"
    elif 110 <= neck_angle < 130:
        status = "Alarming"
        feedback.append("Head slightly forward")
    else:
        status = "Danger"
        feedback.append("Severe forward head posture")

    if shoulder_tilt > 20:
        feedback.append("Uneven shoulders")

    return status, neck_angle, feedback

def log_posture(status, angle, feedback, log_path="log_data.csv"):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    entry = pd.DataFrame([[timestamp, status, angle, "; ".join(feedback)]],
                         columns=["timestamp", "status", "angle", "feedback"])
    entry.to_csv(log_path, mode='a', header=not os.path.exists(log_path), index=False)
