# collect_data.py
"""
Collect sign-language gesture data using MediaPipe Holistic and save as .npy keypoint arrays.

Usage examples:
---------------
# Collect 60 sequences for every gesture listed in config.py
python collect_data.py --samples 60

# Collect 100 additional sequences only for the "None" gesture
python collect_data.py --samples 100 --only None

Notes:
------
• Each gesture sequence = SEQ_LEN frames (default 30) saved under MP_Data/<gesture>/<sequence>/<frame>.npy
• Existing data is never overwritten – sequence numbering auto-continues.
• Press 'q' anytime to quit safely.
"""

import argparse
import time
import cv2
import numpy as np
from pathlib import Path
from config import ACTIONS, SEQ_LEN, DATA_DIR
from keypoints import mp_holistic, mediapipe_detect, draw_landmarks, extract_keypoints


# ---------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--samples", type=int, default=60,
                    help="Number of new sequences to record per gesture")
parser.add_argument("--only", type=str, default=None,
                    help="Collect data only for this gesture name (must match config.ACTIONS)")
args = parser.parse_args()


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------
def ensure_dirs():
    """Make sure root gesture folders exist."""
    for action in ACTIONS:
        (DATA_DIR / action).mkdir(parents=True, exist_ok=True)


def next_sequence_index(action_dir: Path) -> int:
    """Return the next available sequence index to avoid overwriting."""
    existing = [int(p.name) for p in action_dir.glob("*")
                if p.is_dir() and p.name.isdigit()]
    return max(existing) + 1 if existing else 0


# ---------------------------------------------------------------------
# Main routine
# ---------------------------------------------------------------------
def main():
    ensure_dirs()

    gesture_list = [args.only] if args.only else ACTIONS
    cap = cv2.VideoCapture(0)

    with mp_holistic.Holistic(min_detection_confidence=0.6,
                              min_tracking_confidence=0.6) as holistic:

        for gi, action in enumerate(gesture_list):
            action_dir = DATA_DIR / action
            start_idx = next_sequence_index(action_dir)
            total_new = args.samples

            print(f"\n[{gi+1}/{len(gesture_list)}] Collecting gesture: '{action}' "
                  f"(starting at sequence {start_idx}, adding {total_new})")

            for sequence in range(start_idx, start_idx + total_new):
                seq_dir = action_dir / str(sequence)
                seq_dir.mkdir(parents=True, exist_ok=True)

                # Countdown before each new sequence
                for c in range(3, 0, -1):
                    ret, frame = cap.read()
                    if not ret:
                        continue
                    cv2.putText(frame,
                                f"Get ready: '{action}' seq {sequence} in {c}",
                                (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                                (0, 255, 0), 2)
                    cv2.imshow("Collect", frame)
                    if cv2.waitKey(500) & 0xFF == ord('q'):
                        cap.release(); cv2.destroyAllWindows(); return

                print(f"    Sequence {sequence - start_idx + 1}/{total_new}")

                # Record SEQ_LEN frames
                for frame_num in range(SEQ_LEN):
                    ret, frame = cap.read()
                    if not ret:
                        print("Frame grab failed, skipping...")
                        continue

                    image, results = mediapipe_detect(frame, holistic)
                    draw_landmarks(image, results)
                    keypoints = extract_keypoints(results)
                    np.save(seq_dir / f"{frame_num}.npy", keypoints)

                    cv2.putText(image,
                                f"{action} | seq {sequence} | frame {frame_num}",
                                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                                (255, 255, 255), 2)
                    cv2.imshow("Collect", image)

                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        cap.release(); cv2.destroyAllWindows(); return

    cap.release()
    cv2.destroyAllWindows()
    print("\n✅ Data collection complete.")


# ---------------------------------------------------------------------
if __name__ == "__main__":
    main()
