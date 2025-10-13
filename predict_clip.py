# predict_clip.py
import sys
import numpy as np
import tensorflow as tf
from pathlib import Path
from config import SEQ_LEN, MODELS_DIR

"""
Usage:
python predict_clip.py MP_Data/hello/0
"""

def main():
    if len(sys.argv) < 2:
        print("Please provide a sequence folder, e.g., MP_Data/hello/0")
        return

    seq_path = Path(sys.argv[1])
    frames = []
    for f in range(SEQ_LEN):
        frames.append(np.load(seq_path / f"{f}.npy"))
    X = np.expand_dims(np.array(frames, dtype=np.float32), axis=0)  # (1, SEQ_LEN, INPUT_DIM)

    model = tf.keras.models.load_model(MODELS_DIR / "signlstm.h5")
    actions = np.load(MODELS_DIR / "label_map.npy", allow_pickle=True)
    probs = model.predict(X)[0]
    pred_idx = int(probs.argmax())
    print("Predicted:", actions[pred_idx], " | probs:", probs)

if __name__ == "__main__":
    main()
