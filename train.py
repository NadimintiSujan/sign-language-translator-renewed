# train.py
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Masking
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping
from config import ACTIONS, SEQ_LEN, DATA_DIR, MODELS_DIR, RUNS_DIR, INPUT_DIM

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

    X = np.array(X, dtype=np.float32)              # (N, SEQ_LEN, INPUT_DIM)
    y = tf.keras.utils.to_categorical(np.array(y), num_classes=len(ACTIONS))
    return X, y

def build_model(input_dim, seq_len, num_classes):
    model = Sequential([
        Masking(mask_value=0., input_shape=(seq_len, input_dim)),
        LSTM(128, return_sequences=True),
        Dropout(0.3),
        LSTM(128),
        Dropout(0.3),
        Dense(128, activation="relu"),
        Dropout(0.3),
        Dense(num_classes, activation="softmax"),
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
                  loss="categorical_crossentropy",
                  metrics=["accuracy"])
    return model

def main():
    X, y = load_dataset()
    print("Dataset:", X.shape, y.shape)  # (N, 30, INPUT_DIM), (N, 7)

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y.argmax(axis=1)
    )

    model = build_model(INPUT_DIM, SEQ_LEN, len(ACTIONS))

    tb = TensorBoard(log_dir=str(RUNS_DIR / "signlstm"))
    ckpt = ModelCheckpoint(str(MODELS_DIR / "signlstm.best.h5"),
                           monitor="val_accuracy", save_best_only=True, verbose=1)
    es = EarlyStopping(monitor="val_accuracy", patience=15, restore_best_weights=True)

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=100,
        batch_size=32,
        callbacks=[tb, ckpt, es],
        verbose=1
    )

    # Final save (best already saved by ckpt)
    model.save(MODELS_DIR / "signlstm.h5")
    np.save(MODELS_DIR / "label_map.npy", np.array(ACTIONS))
    print("Saved model to models/signlstm.h5 and label_map.npy")

    # Quick val eval
    y_pred = model.predict(X_val).argmax(axis=1)
    y_true = y_val.argmax(axis=1)
    print("Val accuracy:", accuracy_score(y_true, y_pred))
    print(classification_report(y_true, y_pred, target_names=ACTIONS))

if __name__ == "__main__":
    main()
