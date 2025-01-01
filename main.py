"""
main.py

Streamlit application that:
1. Initializes a webcam feed via OpenCV.
2. Draws on a canvas when the user raises their index finger.
3. Clears the canvas (and resets recognized expression/result) when the user raises their thumb.
4. Solves the drawn math expression (OCR + Sympy) when all fingers except pinky are raised.

Dependencies:
- OpenCV for video capture and image manipulation.
- Numpy for image arrays.
- Streamlit for the user interface.
- gesture_utils.py for Mediapipe-based gesture detection.
- solver.py for OCR + math expression evaluation.
"""

import os
import cv2
import numpy as np
import streamlit as st
from PIL import Image
from dotenv import load_dotenv

from gesture_utils import find_hands, fingers_up, draw_on_canvas
from solver import solve_expression_from_canvas

# Load environment variables if needed (e.g., from .env)
load_dotenv()

# Streamlit page configuration
st.set_page_config(page_title="Hand Gesture Solver", page_icon="🖐️", layout="wide")
st.title("Hand Gesture Solver")

# Sidebar instructions for user clarity
st.sidebar.header("Instructions")
st.sidebar.info("""
Hand gestures:
- **Index finger up**: Draw on canvas
- **Thumb up**: Clear canvas & reset expression/result
- **All fingers up except pinky**: Solve the expression from the drawn image
""")

# Initialize Streamlit session states for recognized expressions/results
if "recognized_expr" not in st.session_state:
    st.session_state["recognized_expr"] = ""
if "evaluation_result" not in st.session_state:
    st.session_state["evaluation_result"] = ""

# Set up the webcam
cap = cv2.VideoCapture(0)
cap.set(3, 1280)  # Frame width
cap.set(4, 720)   # Frame height

# Create a blank black canvas for drawing
canvas = np.zeros((720, 1280, 3), dtype=np.uint8)
prev_pos = None  # Track previous position on the canvas for drawing lines

# Streamlit layout: two columns
col1, col2 = st.columns([3, 2])

# Left column: Show the webcam feed with the drawn overlay
with col1:
    # Initialize an image placeholder in Streamlit
    FRAME_WINDOW = st.image(canvas, caption="Webcam Feed", use_container_width=True)

# Right column: Show recognized expression and evaluation result
with col2:
    st.subheader("Recognized Expression")
    expr_placeholder = st.empty()  # A Streamlit placeholder for dynamic text
    st.subheader("Result")
    result_placeholder = st.empty()

# Main loop: read frames from the webcam
while True:
    success, img = cap.read()
    if not success:
        # If webcam can't be read, show a warning and break the loop
        st.warning("Could not read webcam. Ensure it's connected, then refresh.")
        break

    # Flip the image horizontally (mirror-like experience)
    img = cv2.flip(img, 1)

    # Detect hands in the current frame
    hands_info, img = find_hands(img, draw=False)

    # If at least one hand is found, process gestures
    if hands_info:
        # Take the first detected hand
        hand = hands_info[0]
        # Determine which fingers are raised
        finger_list = fingers_up(hand)
        # Update the drawing canvas based on the gesture
        prev_pos, canvas = draw_on_canvas(finger_list, hand["lmList"], prev_pos, canvas)

        # If all fingers except pinky are up => Solve the expression on the canvas
        if finger_list == [1, 1, 1, 1, 0]:
            # Solve expression: OCR + cleaning + Sympy evaluation
            st.session_state["recognized_expr"], st.session_state["evaluation_result"] = solve_expression_from_canvas(canvas)

    # Blend the webcam feed with the drawing overlay
    combined = cv2.addWeighted(img, 0.7, canvas, 0.3, 0)
    # Update the Streamlit image display
    FRAME_WINDOW.image(combined, channels="BGR")

    # Display recognized expression and result in real time
    expr_placeholder.text(st.session_state["recognized_expr"])
    result_placeholder.text(st.session_state["evaluation_result"]) 