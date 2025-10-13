# realtime.py
import cv2
import numpy as np
import tensorflow as tf
from collections import deque
from config import ACTIONS, SEQ_LEN, MODELS_DIR
from keypoints import mp_holistic, mediapipe_detect, draw_landmarks, extract_keypoints

def main():
    model = tf.keras.models.load_model(MODELS_DIR / "signlstm.h5")
    actions = np.load(MODELS_DIR / "label_map.npy", allow_pickle=True)

    cap = cv2.VideoCapture(0)
    sequence = deque(maxlen=SEQ_LEN)
    last_label = ""
    stable_counter = 0

    with mp_holistic.Holistic(min_detection_confidence=0.6, min_tracking_confidence=0.6) as holistic:
        while True:
            ret, frame = cap.read()
            if not ret: break
            image, results = mediapipe_detect(frame, holistic)
            draw_landmarks(image, results)
            keypoints = extract_keypoints(results)
            sequence.append(keypoints)

            if len(sequence) == SEQ_LEN:
                X = np.expand_dims(np.array(sequence, dtype=np.float32), axis=0)
                probs = model.predict(X, verbose=0)[0]
                pred_idx = int(np.argmax(probs))
                label = actions[pred_idx]
                conf = float(np.max(probs))

                if label == last_label:
                    stable_counter += 1
                else:
                    stable_counter = 0
                    last_label = label

                # Display
                text = f"{label} ({conf:.2f})"
                color = (0, 255, 0) if conf >= 0.7 and stable_counter > 3 else (0, 200, 255)
                cv2.rectangle(image, (0,0), (300, 50), (0,0,0), -1)
                cv2.putText(image, text, (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

            cv2.imshow("Real-time Sign Inference", image)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
