# evaluate.py
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import tensorflow as tf
from config import ACTIONS, SEQ_LEN, DATA_DIR, MODELS_DIR, INPUT_DIM

def load_dataset():
    X, y = [], []
    label_map = {action: i for i, action in enumerate(ACTIONS)}
    for action in ACTIONS:
        action_dir = DATA_DIR / action
        if not action_dir.exists(): continue
        for seq_dir in sorted(action_dir.iterdir()):
            frames = []
            for f in range(SEQ_LEN):
                path = seq_dir / f"{f}.npy"
                if not path.exists():
                    frames = []
                    break
                frames.append(np.load(path))
            if len(frames) == SEQ_LEN:
                X.append(frames)
                y.append(label_map[action])

    X = np.array(X, dtype=np.float32)
    y = tf.keras.utils.to_categorical(np.array(y), num_classes=len(ACTIONS))
    return X, y

def main():
    X, y = load_dataset()
    model = tf.keras.models.load_model(MODELS_DIR / "signlstm.h5")
    print("Loaded model.")

    loss, acc = model.evaluate(X, y, verbose=0)
    print(f"Overall loss={loss:.4f} acc={acc:.4f}")

    y_pred = model.predict(X).argmax(axis=1)
    y_true = y.argmax(axis=1)

    print("Confusion matrix:\n", confusion_matrix(y_true, y_pred))
    print(classification_report(y_true, y_pred, target_names=ACTIONS))

if __name__ == "__main__":
    main()
