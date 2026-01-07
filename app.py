import streamlit as st
import cv2
import numpy as np
import mediapipe as mp
from datetime import date
from PIL import Image

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="ACL-Patch", 
    page_icon="🩹", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- HELPER FUNCTIONS ---
def calculate_angle(a, b, c):
    """Calculates angle between three joints."""
    a = np.array(a) # First
    b = np.array(b) # Mid
    c = np.array(c) # End
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)
    if angle > 180.0: angle = 360-angle
    return angle

# --- PAGE 1: HOME DASHBOARD ---
def show_home():
    st.title("🩹 ACL-Patch Dashboard")
    st.markdown("### Status: Pre-Operative Phase")
    
    # Surgery Countdown Logic
    surgery_date = date(2026, 1, 31) # Updated to your year (assuming 2026 based on prompt)
    today = date.today()
    days_left = (surgery_date - today).days
    
    # Metrics Row
    col1, col2, col3 = st.columns(3)
    col1.metric("Surgery Countdown", f"{days_left} Days", "Jan 31")
    col2.metric("Current Condition", "Prehab", "Maintain Strength")
    col3.metric("System Status", "Vision Online", "AI Offline")

    st.markdown("---")
    st.info("ℹ️ **Daily Task:** Please record your baseline measurements in the 'Vision System' tab.")

# --- PAGE 2: VISION SYSTEM (Upload Mode) ---
def show_vision():
    st.title("👁️ Bio-Mechanics Scanner")
    st.markdown("**Mode:** Static Image Analysis")
    st.info("Upload a photo of your movement (Side View) to measure knee angles.")

    # 1. File Uploader
    uploaded_file = st.file_uploader("Upload Image", type=['jpg', 'jpeg', 'png'])

    if uploaded_file is not None:
        # Load the image
        image = Image.open(uploaded_file)
        image_np = np.array(image) # Convert to numpy for MediaPipe

        # Create 2 columns: Original vs Analyzed
        col1, col2 = st.columns(2)
        with col1:
            st.image(image, caption="Original Photo", use_container_width=True)

        # 2. Process with MediaPipe
        mp_pose = mp.solutions.pose
        
        # static_image_mode=True is more accurate for photos
        with mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5) as pose:
            results = pose.process(image_np)
            
            # Create a copy to draw on
            annotated_image = image_np.copy()

            if results.pose_landmarks:
                # Draw the skeleton
                mp.solutions.drawing_utils.draw_landmarks(
                    annotated_image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
                
                # Extract Landmarks
                try:
                    landmarks = results.pose_landmarks.landmark
                    
                    # Get coordinates (Left Leg)
                    hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x, landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
                    knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x, landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
                    ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x, landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]
                    
                    # Calculate Angle
                    angle = calculate_angle(hip, knee, ankle)
                    
                    # Display Result in Column 2
                    with col2:
                        st.image(annotated_image, caption=f"Analyzed: {int(angle)}° Knee Flexion", use_container_width=True)
                        
                        # Data Card
                        if angle > 170:
                            st.error(f"⚠️ **{int(angle)}°** - Hyperextension Risk")
                        elif angle < 45:
                            st.warning(f"⚠️ **{int(angle)}°** - Deep Flexion (Caution)")
                        else:
                            st.success(f"✅ **{int(angle)}°** - Safe Range")
                            
                except Exception as e:
                    st.error(f"Could not calculate angle. Ensure full body is visible. ({e})")
            else:
                with col2:
                    st.warning("⚠️ No human detected in the image.")

# --- PAGE 3: AI COACH (The Brain) ---
def show_coach():
    st.title("🧠 Protocol Advisor")
    st.caption("Powered by OpenAI (Currently Offline Mode)")
    
    # Simple Chat Layout
    st.markdown("Ask questions about your **Pre-Surgery Protocol**.")
    
    messages = st.container()
    user_input = st.text_input("Your Question:", placeholder="e.g., Can I take Ibuprofen today?")
    
    if st.button("Send Query"):
        # Placeholder logic since Internet is down
        with messages:
            st.chat_message("user").write(user_input)
            st.chat_message("assistant").write("⚠️ **Network Error:** I cannot reach the cloud server right now. \n\n*Safety Tip: If in doubt, follow the RICE method (Rest, Ice, Compression, Elevation).*")

# --- MAIN NAVIGATION CONTROLLER ---
def main():
    # Sidebar Navigation
    st.sidebar.title("Navigation")
    selection = st.sidebar.radio("Go to:", ["🏠 Home", "👁️ Vision System", "🧠 AI Coach"])

    st.sidebar.markdown("---")
    st.sidebar.info("Project: ACL-Patch\nVer: 1.0.0 (Pre-Op)")

    # Page Routing
    if selection == "🏠 Home":
        show_home()
    elif selection == "👁️ Vision System":
        show_vision()
    elif selection == "🧠 AI Coach":
        show_coach()

if __name__ == "__main__":
    main()