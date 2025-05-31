import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
import av
import cv2
import numpy as np
import mediapipe as mp
from posture_utils import analyze_posture, log_posture
import pandas as pd
import matplotlib.pyplot as plt
import os

# Initialize MediaPipe Pose
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Create CSV log file if it doesn't exist
if not os.path.exists("log_data.csv"):
    pd.DataFrame(columns=["timestamp", "status", "angle", "feedback"]).to_csv("log_data.csv", index=False)

# Define the Video Processor class
class PostureProcessor(VideoProcessorBase):
    def __init__(self):
        self.last_status = ""

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        image = frame.to_ndarray(format="bgr24")
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = pose.process(rgb_image)

        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            status, angle, feedback = analyze_posture(landmarks)

            # Draw landmarks
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

            # Overlay feedback
            color = (0, 255, 0) if status == "Good" else (0, 255, 255) if status == "Alarming" else (0, 0, 255)
            cv2.putText(image, f"Status: {status}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
            for i, tip in enumerate(feedback):
                cv2.putText(image, tip, (10, 60 + i * 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

            # Log posture
            log_posture(status, angle, feedback)

        return av.VideoFrame.from_ndarray(image, format="bgr24")

# Streamlit UI
st.set_page_config(page_title="Real-Time Posture Monitor", layout="wide")
st.title("🧍 Real-Time Posture Monitoring App")
st.markdown("Monitor your posture live using your webcam. Get real-time feedback and analytics.")

# Webcam Stream
webrtc_streamer(key="posture", video_processor_factory=PostureProcessor)

# Display Log Data
if os.path.exists("log_data.csv"):
    df = pd.read_csv("log_data.csv")
    st.subheader("📈 Posture Angle Trend")
    fig, ax = plt.subplots()
    df["angle"] = pd.to_numeric(df["angle"], errors='coerce')
    df.dropna(subset=["angle"], inplace=True)
    ax.plot(df["timestamp"], df["angle"], label="Angle")
    ax.set_ylabel("Angle (degrees)")
    ax.set_xlabel("Time")
    ax.tick_params(axis='x', rotation=45)
    ax.set_title("Posture Angle Over Time")
    st.pyplot(fig)

    st.subheader("📝 Summary")
    st.write(df.tail(5))
