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

# --- PAGE 2: VISION SYSTEM (Updated for Accuracy) ---
def show_vision():
    st.title("👁️ Bio-Mechanics Scanner")
    
    # 1. NEW: Settings Sidebar for this page
    col_settings, col_display = st.columns([1, 3])
    
    with col_settings:
        st.markdown("### ⚙️ Scanner Settings")
        # Toggle: Which leg to track?
        target_side = st.radio("Select Leg to Track:", ["Left Leg", "Right Leg"], index=0)
        
        # Toggle: How to display 0 degrees?
        # Clinical: 0° is straight. Geometric: 180° is straight.
        mode = st.radio("Angle Mode:", ["Geometric (180° = Straight)", "Clinical (0° = Straight)"], index=0)
        
        uploaded_file = st.file_uploader("Upload Image", type=['jpg', 'jpeg', 'png'])

    # 2. Process Image
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        image_np = np.array(image)

        # Initialize MediaPipe
        mp_pose = mp.solutions.pose
        with mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5) as pose:
            results = pose.process(image_np)
            annotated_image = image_np.copy()

            if results.pose_landmarks:
                landmarks = results.pose_landmarks.landmark
                
                # --- NEW: Dynamic Leg Selection ---
                if target_side == "Left Leg":
                    # Get Left coordinates
                    a = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x, landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
                    b = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x, landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
                    c = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x, landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]
                    color = (255, 0, 0) # Red color for Left
                else:
                    # Get Right coordinates
                    a = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
                    b = [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y]
                    c = [landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y]
                    color = (0, 0, 255) # Blue color for Right
                
                # Calculate Angle
                angle = calculate_angle(a, b, c)

                # --- NEW: Clinical Conversion ---
                display_angle = angle
                if mode == "Clinical (0° = Straight)":
                    display_angle = abs(180 - angle)

                # Visuals
                with col_display:
                    # Draw the specific triangle we are measuring
                    h, w, _ = annotated_image.shape
                    
                    # Draw Lines (Thick)
                    cv2.line(annotated_image, tuple(np.multiply(a, [w, h]).astype(int)), tuple(np.multiply(b, [w, h]).astype(int)), color, 5)
                    cv2.line(annotated_image, tuple(np.multiply(b, [w, h]).astype(int)), tuple(np.multiply(c, [w, h]).astype(int)), color, 5)
                    
                    # Draw Circle at Knee
                    cv2.circle(annotated_image, tuple(np.multiply(b, [w, h]).astype(int)), 10, (255, 255, 255), -1)

                    st.image(annotated_image, caption=f"Analyzed: {target_side}", use_container_width=True)
                    
                    # Result Card
                    st.metric(label=f"{target_side} Angle", value=f"{int(display_angle)}°")
                    
                    if angle > 165: # Geometric Straight
                        st.success("Leg is Fully Extended (Safe)")
                    elif angle < 60:
                        st.warning("Deep Flexion Detected")
            else:
                st.error("No pose detected. Try a clear side-profile photo.")

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