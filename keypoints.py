# keypoints.py
import numpy as np
import mediapipe as mp
import cv2
from config import USE_FACE, USE_POSE, USE_LEFT_HAND, USE_RIGHT_HAND, FACE_LM, POSE_LM, HAND_LM

mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

def mediapipe_detect(image, holistic):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = holistic.process(image)
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image, results

def draw_landmarks(image, results):
    # Optional visualize; not required for saving npy
    if results.pose_landmarks:
        mp_drawing.draw_landmarks(image, results.pose_landmarks,
                                  mp_holistic.POSE_CONNECTIONS)
    if results.left_hand_landmarks:
        mp_drawing.draw_landmarks(image, results.left_hand_landmarks,
                                  mp_holistic.HAND_CONNECTIONS)
    if results.right_hand_landmarks:
        mp_drawing.draw_landmarks(image, results.right_hand_landmarks,
                                  mp_holistic.HAND_CONNECTIONS)
    if results.face_landmarks:
        mp_drawing.draw_landmarks(image, results.face_landmarks,
                                  mp_holistic.FACEMESH_TESSELATION)

def extract_keypoints(results):
    # Face
    if USE_FACE:
        if results.face_landmarks:
            face = np.array([[lm.x, lm.y, lm.z] for lm in results.face_landmarks.landmark]).flatten()
        else:
            face = np.zeros(FACE_LM)
    else:
        face = np.array([])
    # Pose
    if USE_POSE:
        if results.pose_landmarks:
            pose = np.array([[lm.x, lm.y, lm.z, lm.visibility] for lm in results.pose_landmarks.landmark]).flatten()
        else:
            pose = np.zeros(POSE_LM)
    else:
        pose = np.array([])
    # Hands
    if USE_LEFT_HAND:
        if results.left_hand_landmarks:
            lhand = np.array([[lm.x, lm.y, lm.z] for lm in results.left_hand_landmarks.landmark]).flatten()
        else:
            lhand = np.zeros(HAND_LM)
    else:
        lhand = np.array([])
    if USE_RIGHT_HAND:
        if results.right_hand_landmarks:
            rhand = np.array([[lm.x, lm.y, lm.z] for lm in results.right_hand_landmarks.landmark]).flatten()
        else:
            rhand = np.zeros(HAND_LM)
    else:
        rhand = np.array([])

    return np.concatenate([face, pose, lhand, rhand]).astype(np.float32)
