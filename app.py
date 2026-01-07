import streamlit as st
import cv2
import numpy as np
import mediapipe as mp
from openai import OpenAI
import os
from dotenv import load_dotenv

# 1. Load Environment Variables
load_dotenv()

# 2. Setup Page Config (Updated Name)
st.set_page_config(
    page_title="ACL-Patch: Recovery Engine", 
    page_icon="🩹", 
    layout="wide"
)

# Custom Header
st.title("🩹 ACL-Patch")
st.markdown("**System Status:** Pre-Op Phase | **Surgery Date:** Jan 31")

# 3. Sidebar: The "Prehab Agent"
st.sidebar.header("🧠 Protocol Advisor")
api_key = os.getenv("OPENAI_API_KEY")

if not api_key:
    st.sidebar.error("⚠️ API Key missing in .env file")
else:
    client = OpenAI(api_key=api_key)

    # Simplified Context for Pre-Op
    user_query = st.sidebar.text_input("Consult the Protocol:", "Is it safe to do leg extensions?")

    if st.sidebar.button("Analyze Query"):
        try:
            with st.spinner("Consulting knowledge base..."):
                response = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[
                        {"role": "system", "content": "You are 'ACL-Patch', an expert post-op recovery assistant. Current User Status: 3 weeks pre-op. Focus on inflammation reduction and quad retention. Keep answers under 50 words."},
                        {"role": "user", "content": user_query}
                    ]
                )
                st.sidebar.info(response.choices[0].message.content)
        except Exception as e:
            st.sidebar.error(f"Connection Error: {e}")

# 4. Main Area: Vision System
st.header("👁️ Biometric Scanner")
col1, col2 = st.columns([2, 1])

with col1:
    st.write("Real-time Graft Protection Monitoring")
    run_camera = st.checkbox("Initialize Camera Feed")
    FRAME_WINDOW = st.image([])

with col2:
    st.write("**Live Telemetry**")
    st.metric(label="Knee Angle", value="--°")
    st.metric(label="Status", value="Standby")

# Vision Logic
if run_camera:
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose()
    cap = cv2.VideoCapture(0)

    while run_camera:
        ret, frame = cap.read()
        if not ret:
            st.error("Camera not detected.")
            break
        
        # Processing
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(image)

        if results.pose_landmarks:
            mp.solutions.drawing_utils.draw_landmarks(
                image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            
        FRAME_WINDOW.image(image)
    
    cap.release()