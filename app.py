import streamlit as st
import cv2
import posture_utils as pu
import pandas as pd
import matplotlib.pyplot as plt
import time
import os
import base64




st.set_page_config(page_title="Posture Monitor", layout="wide")
st.title("📹 Real-Time Posture Monitoring")
st.markdown("Track your body posture and get real-time feedback using AI.")

if "monitoring" not in st.session_state:
    st.session_state["monitoring"] = False


log_path = "log_data.csv"
monitoring = st.session_state.get("monitoring", False)

# Button logic
col1, col2 = st.columns([1, 1])
with col1:
    if st.button("▶️ Start / Resume Monitoring"):
        st.session_state.monitoring = True
with col2:
    if st.button("⏸️ Pause Monitoring"):
        st.session_state.monitoring = False

# Load alert sound as base64
def get_audio_base64(file_path):
    with open(file_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode()
    return f'<audio autoplay><source src="data:audio/mp3;base64,{b64}" type="audio/mp3"></audio>'

if st.session_state.monitoring:
    cap = cv2.VideoCapture(0)
    stframe = st.empty()
    status_box = st.empty()

    last_alert_time = 0

    while st.session_state.monitoring:
        ret, frame = cap.read()
        if not ret:
            st.error("Webcam not found.")
            break

        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pu.pose.process(image)

        if results.pose_landmarks:
            pu.mp_drawing.draw_landmarks(image, results.pose_landmarks, pu.mp_pose.POSE_CONNECTIONS)
            landmarks = results.pose_landmarks.landmark

            status, angle, feedback = pu.analyze_posture(landmarks)
            pu.log_posture(status, angle, feedback, log_path)

            # Show feedback
            color = (0, 255, 0) if status == "Good" else (0, 165, 255) if status == "Alarming" else (0, 0, 255)
            cv2.putText(image, f"Posture: {status}", (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)

            # Trigger alert for danger
            if status == "Danger" and time.time() - last_alert_time > 20:
                last_alert_time = time.time()
                st.markdown(get_audio_base64("alarm.mp3"), unsafe_allow_html=True)

        stframe.image(image, channels='RGB')

        # Feedback box
        with status_box:
            st.markdown(f"### Current Status: **{status}**")
            if feedback:
                st.warning(" | ".join(feedback))

    cap.release()

# Trend summary
st.subheader("📊 Posture Trend Summary")
if os.path.exists(log_path):
    df = pd.read_csv(log_path)
    counts = df["status"].value_counts().to_dict()
    total = sum(counts.values()) or 1

    col1, col2, col3 = st.columns(3)
    col1.metric("✅ Good", f"{counts.get('Good', 0)} ({counts.get('Good',0)*100//total}%)")
    col2.metric("⚠️ Alarming", f"{counts.get('Alarming', 0)} ({counts.get('Alarming',0)*100//total}%)")
    col3.metric("❌ Danger", f"{counts.get('Danger', 0)} ({counts.get('Danger',0)*100//total}%)")

    # Line chart
    fig, ax = plt.subplots()
    ax.plot(pd.to_datetime(df["timestamp"]), df["angle"], label="Neck Angle")
    ax.axhspan(95, 110, color='green', alpha=0.2, label="Good Range")
    ax.axhspan(110, 130, color='orange', alpha=0.2, label="Alarming Range")
    ax.axhspan(130, 180, color='red', alpha=0.2, label="Danger Range")
    ax.legend()
    ax.set_title("Neck Angle Over Time")
    ax.set_ylabel("Angle (°)")
    ax.set_xlabel("Time")
    st.pyplot(fig)
else:
    st.info("No data available. Start monitoring to view trends.")
