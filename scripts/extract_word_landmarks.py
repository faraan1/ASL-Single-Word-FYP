import os
import json
import cv2
import numpy as np
import mediapipe as mp

# --- Configuration ---
FILTERED_WORDS_PATH = "data/words/filtered_words.json"
VIDEOS_FOLDER = "data/words/videos"
OUTPUT_DIR = "data/words/processed"
SEQUENCE_LENGTH = 30  # frames per sample

# --- MediaPipe setup (video mode) ---
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# --- Load our filtered word -> video mapping ---
with open(FILTERED_WORDS_PATH, "r") as f:
    filtered_words = json.load(f)

print(f"Loaded {len(filtered_words)} words to process")
for word, instances in filtered_words.items():
    print(f"  '{word}': {len(instances)} video(s)")

def extract_normalized_landmarks(hand_landmarks):
    """Same normalization as the letters branch: wrist-relative + scale-normalized."""
    coords = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark])
    wrist = coords[0]
    coords = coords - wrist
    scale = np.linalg.norm(coords[9])
    if scale > 0:
        coords = coords / scale
    return coords.flatten()


def resample_sequence(frames, target_length):
    """
    Takes a list of per-frame landmark vectors (variable length) and
    resamples it to exactly `target_length` frames by picking evenly
    spaced indices - stretching shorter sequences, trimming longer ones.
    """
    frames = np.array(frames)
    current_length = len(frames)

    if current_length == target_length:
        return frames

    indices = np.linspace(0, current_length - 1, target_length)
    indices = np.round(indices).astype(int)

    return frames[indices]

def process_video(video_path, frame_start=1, frame_end=-1):
    """
    Reads a video, extracts a per-frame landmark vector for each frame
    (within the optional frame_start/frame_end range), and returns the
    resampled fixed-length sequence.
    """
    cap = cv2.VideoCapture(video_path)
    frame_landmarks = []
    frame_num = 0

    while True:
        success, frame = cap.read()
        if not success:
            break

        frame_num += 1
        if frame_num < frame_start:
            continue
        if frame_end != -1 and frame_num > frame_end:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)

        landmarks_vector = np.zeros(126)
        if results.multi_hand_landmarks:
            for hand_index, hand_landmarks in enumerate(results.multi_hand_landmarks):
                if hand_index >= 2:
                    break
                normalized = extract_normalized_landmarks(hand_landmarks)
                start = hand_index * 63
                landmarks_vector[start:start + 63] = normalized

        frame_landmarks.append(landmarks_vector)

    cap.release()

    if len(frame_landmarks) == 0:
        return None

    return resample_sequence(frame_landmarks, SEQUENCE_LENGTH)


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    all_sequences = []
    all_labels = []

    for word, instances in filtered_words.items():
        success_count = 0
        fail_count = 0

        for instance in instances:
            video_id = instance["video_id"]
            frame_start = instance.get("frame_start", 1)
            frame_end = instance.get("frame_end", -1)
            video_path = os.path.join(VIDEOS_FOLDER, f"{video_id}.mp4")

            sequence = process_video(video_path, frame_start, frame_end)

            if sequence is not None:
                all_sequences.append(sequence)
                all_labels.append(word)
                success_count += 1
            else:
                fail_count += 1

        print(f"Word '{word}': {success_count} successful, {fail_count} failed")

    X = np.array(all_sequences)
    y = np.array(all_labels)

    np.save(os.path.join(OUTPUT_DIR, "word_sequences.npy"), X)
    np.save(os.path.join(OUTPUT_DIR, "word_labels.npy"), y)

    print(f"\nTotal sequences: {len(X)}")
    print(f"Sequence shape: {X.shape}")
    print(f"Saved to {OUTPUT_DIR}/word_sequences.npy and word_labels.npy")


if __name__ == "__main__":
    main()

