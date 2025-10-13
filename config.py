# config.py
from pathlib import Path

# Your 7 gestures — keep this EXACT ordering consistent everywhere
ACTIONS = [
    "None",
    "hello",
    "thank you",
    "eat",
    "help",
    "excuse me",
    "please",
]

# Each training sample is a sequence of N frames (recommended: 30)
SEQ_LEN = 30

# Root where .npy frame files per sequence are stored
DATA_DIR = Path("MP_Data")

# Model & artifacts
MODELS_DIR = Path("models")
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# TensorBoard runs
RUNS_DIR = Path("runs")
RUNS_DIR.mkdir(parents=True, exist_ok=True)

# Choose which landmarks to use (pose is most stable; hands add signal if you sign with hands)
USE_FACE = False           # 468*3
USE_POSE = True            # 33*4 (x,y,z,visibility)
USE_LEFT_HAND = True       # 21*3
USE_RIGHT_HAND = True      # 21*3

# Derived feature vector length
FACE_LM = 468 * 3
POSE_LM = 33 * 4
HAND_LM = 21 * 3

def vector_len():
    total = 0
    if USE_FACE: total += FACE_LM
    if USE_POSE: total += POSE_LM
    if USE_LEFT_HAND: total += HAND_LM
    if USE_RIGHT_HAND: total += HAND_LM
    return total

INPUT_DIM = vector_len()   # per-frame feature size
