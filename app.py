import streamlit as st
import cv2
import numpy as np
import mediapipe as mp
# We commented out OpenAI for now so you can work OFFLINE
# from openai import OpenAI 
# from dotenv import load_dotenv

# load_dotenv()

st.set_page_config(page_title="ACL-Patch: Vision Mode", page_icon="🩹", layout="wide")

# --- HELPER FUNCTION: Calculate Angle ---
def calculate_angle(a, b, c):
    """
    Calculates the angle at point b given points a, b, c.
    Arguments are [x, y] coordinates.
    """
    a = np.array(a) # Hip
    b = np.array(b) # Knee
    c = np.array(c) # Ankle
    
    # Calculate arctan2 (returns radians)
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)
    
    if angle > 180.0:
        angle = 360-angle
        
    return angle

st.title("🩹 ACL-Patch: Offline Vision Mode")
st.warning("⚠️ Internet Issue Detected: OpenAI Module Paused. Running in Offline Vision Mode.")

col1, col2 = st.columns([3, 1])

with col1:
    run_camera = st.checkbox("Start Bio-Mechanics Scan")
    FRAME_WINDOW = st.image([])

with col2:
    st.write("## Real-Time Telemetry")
    angle_metric = st.empty()
    status_metric = st.empty()
    st.markdown("---")
    st.write("**Target Range:** 0° - 130°")

if run_camera:
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
    cap = cv2.VideoCapture(0)

    while run_camera:
        ret, frame = cap.read()
        if not ret:
            st.error("No camera found.")
            break
            
        # Recolor to RGB
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = pose.process(image)
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        # Extract Landmarks
        try:
            landmarks = results.pose_landmarks.landmark
            
            # Get coordinates for LEFT leg (Change to RIGHT_HIP etc. if needed)
            hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x, landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
            knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x, landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
            ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x, landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]
            
            # Calculate Angle
            angle = calculate_angle(hip, knee, ankle)
            
            # Update Dashboard
            angle_metric.metric("Knee Angle", f"{int(angle)}°")
            
            # Visual Logic (Safety Check)
            if angle > 170:
                status_metric.error("⚠️ HYPER EXTENSION")
                color = (0, 0, 255) # Red
            elif angle < 45:
                status_metric.warning("⚠️ DEEP FLEXION")
                color = (0, 165, 255) # Orange
            else:
                status_metric.success("✅ SAFE ZONE")
                color = (255, 255, 255) # White
                
            # Render Angle on Video
            cv2.putText(image, str(int(angle)), 
                           tuple(np.multiply(knee, [640, 480]).astype(int)), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2, cv2.LINE_AA
                        )
            
        except:
            pass # Pose not detected

        # Draw stick figure
        if results.pose_landmarks:
            mp.solutions.drawing_utils.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        # Convert back to RGB for Streamlit
        FRAME_WINDOW.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    cap.release()