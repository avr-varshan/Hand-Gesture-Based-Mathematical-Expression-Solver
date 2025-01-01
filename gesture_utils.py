"""
gesture_utils.py

Handles:
1. Hand detection using Mediapipe.
2. Determining which fingers are raised (fingers_up).
3. Drawing or clearing the canvas based on gestures.

Dependencies:
- mediapipe
- opencv (for drawing lines on the canvas)
- numpy
- streamlit (for clearing recognized expressions/results in session state)
"""

import cv2
import numpy as np
import mediapipe as mp
import streamlit as st

# Mediapipe Hands setup
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    model_complexity=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5
)

# Landmark indices for fingertips in Mediapipe's hand model
tip_ids = [4, 8, 12, 16, 20]

def find_hands(img, draw=False, flip_type=True):
    """
    Detect hands in a BGR image using Mediapipe.

    Args:
        img (numpy.ndarray): BGR image from the webcam.
        draw (bool): If True, draw hand landmarks on the image.
        flip_type (bool): Flip 'Right' -> 'Left' label if needed.

    Returns:
        tuple: (all_hands, annotated_img)
        - all_hands: list of dicts, each having:
            {
              'lmList': [[x, y, z], ...],
              'bbox': (xmin, ymin, width, height),
              'center': (cx, cy),
              'type': 'Left'/'Right'
            }
        - annotated_img: the image with optional landmarks drawn
    """
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)
    all_hands = []
    h, w, c = img.shape

    if results.multi_hand_landmarks and results.multi_handedness:
        for hand_type, hand_landmarks in zip(
            results.multi_handedness,
            results.multi_hand_landmarks
        ):
            my_hand = {}
            lm_list = []
            x_list, y_list = [], []

            for lm in hand_landmarks.landmark:
                px, py, pz = int(lm.x * w), int(lm.y * h), lm.z * w
                lm_list.append([px, py, pz])
                x_list.append(px)
                y_list.append(py)

            xmin, xmax = min(x_list), max(x_list)
            ymin, ymax = min(y_list), max(y_list)
            bbox = (xmin, ymin, xmax - xmin, ymax - ymin)
            cx, cy = (xmin + xmax) // 2, (ymin + ymax) // 2

            my_hand["lmList"] = lm_list
            my_hand["bbox"] = bbox
            my_hand["center"] = (cx, cy)

            # Flip type if needed
            if flip_type and hand_type.classification[0].label == "Right":
                my_hand["type"] = "Left"
            else:
                my_hand["type"] = "Right"

            all_hands.append(my_hand)

            if draw:
                mp_draw.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    return all_hands, img

def fingers_up(my_hand):
    """
    Determine which fingers are raised.

    Args:
        my_hand (dict): with keys like 'lmList', 'type', etc.

    Returns:
        list of int: [thumb, index, middle, ring, pinky] => 1 if up, else 0
    """
    lm_list = my_hand["lmList"]
    # If 'type' is 'Right', the thumb logic is reversed horizontally
    hand_type = my_hand.get("type", "Right")
    fingers = []

    # Thumb
    if hand_type == "Right":
        # Compare x of thumb tip vs x of (thumb tip - 1)
        fingers.append(1 if lm_list[tip_ids[0]][0] > lm_list[tip_ids[0] - 1][0] else 0)
    else:
        fingers.append(1 if lm_list[tip_ids[0]][0] < lm_list[tip_ids[0] - 1][0] else 0)

    # Other four fingers
    for i in range(1, 5):
        fingers.append(1 if lm_list[tip_ids[i]][1] < lm_list[tip_ids[i] - 2][1] else 0)

    return fingers

def draw_on_canvas(fingers, lm_list, prev_pos, canvas):
    """
    Draw on the canvas or clear it based on the gesture.

    Gestures:
    - Index finger up ([0,1,0,0,0]) => draw on the canvas
    - Thumb up ([1,0,0,0,0]) => clear the canvas & reset recognized expression/result

    Args:
        fingers (list[int]): finger states [thumb, index, middle, ring, pinky]
        lm_list (list[list[int|float]]): landmarks with x,y,z
        prev_pos (tuple|None): last position on the canvas
        canvas (np.ndarray): the drawing canvas

    Returns:
        tuple: (updated_prev_pos, updated_canvas)
    """
    current_pos = None

    # If index finger is up => draw
    if fingers == [0, 1, 0, 0, 0]:
        current_pos = lm_list[8][0:2]  # index tip's x,y
        if prev_pos is None:
            prev_pos = current_pos
        cv2.line(canvas, tuple(prev_pos), tuple(current_pos), (255, 0, 255), 10)
        prev_pos = current_pos

    # If thumb is up => clear canvas
    elif fingers == [1, 0, 0, 0, 0]:
        canvas = np.zeros_like(canvas)
        prev_pos = None
        # Clear recognized expression/result from session state
        st.session_state["recognized_expr"] = ""
        st.session_state["evaluation_result"] = ""

    else:
        # No drawing or clearing in other gestures
        prev_pos = None

    return prev_pos, canvas