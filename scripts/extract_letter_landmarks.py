import os
import random
import numpy as np
import pandas as pd
import cv2
import mediapipe as mp

# --- Configuration ---
TRAIN_DIR = "data/letters/asl_alphabet_train/asl_alphabet_train"
OUTPUT_DIR = "data/letters/processed"
SAMPLES_PER_LETTER = 200

# Letters we're using (excludes J and Z, which need motion)
LETTERS = [chr(c) for c in range(ord('A'), ord('Z') + 1) if chr(c) not in ('J', 'Z')]

# MediaPipe Hands setup - static_image_mode=True since we're processing photos, not video
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=True,
    max_num_hands=2,
    min_detection_confidence=0.5
)

def extract_landmarks_from_image(image_path):
    """
    Reads one image, runs MediaPipe hand detection, and returns
    a 126-length feature vector (2 hands x 21 landmarks x 3 coords).
    Returns None if no hand was detected.
    """
    image = cv2.imread(image_path)
    if image is None:
        return None

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(image_rgb)

    # Start with an all-zero vector for 2 hands worth of landmarks
    landmarks_vector = np.zeros(126)

    if results.multi_hand_landmarks:
        for hand_index, hand_landmarks in enumerate(results.multi_hand_landmarks):
            if hand_index >= 2:
                break  # only handle up to 2 hands

            coords = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark])

            # Normalize: make the wrist (landmark 0) the origin
            wrist = coords[0]
            coords = coords - wrist

            # Normalize: scale relative to wrist-to-middle-finger-base distance
            # (landmark 9 = base of middle finger - a stable reference size for the hand)
            scale = np.linalg.norm(coords[9])
            if scale > 0:
                coords = coords / scale

            start = hand_index * 63
            landmarks_vector[start:start + 63] = coords.flatten()

        return landmarks_vector
    else:
        return None  # no hand detected in this image

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    all_features = []
    all_labels = []

    for letter in LETTERS:
        letter_folder = os.path.join(TRAIN_DIR, letter)
        all_images = os.listdir(letter_folder)

        # Shuffle ALL images for this letter, then keep going until we hit
        # our target successful count (or run out of images)
        random.shuffle(all_images)

        success_count = 0
        fail_count = 0

        for image_name in all_images:
            if success_count >= SAMPLES_PER_LETTER:
                break  # we've got enough good samples for this letter

            image_path = os.path.join(letter_folder, image_name)
            landmarks = extract_landmarks_from_image(image_path)

            if landmarks is not None:
                all_features.append(landmarks)
                all_labels.append(letter)
                success_count += 1
            else:
                fail_count += 1

        print(f"Letter '{letter}': {success_count} successful, {fail_count} failed (no hand detected)")

    # Convert to a DataFrame and save
    df = pd.DataFrame(all_features)
    df["label"] = all_labels

    output_path = os.path.join(OUTPUT_DIR, "letter_landmarks.csv")
    df.to_csv(output_path, index=False)

    print(f"\nTotal samples: {len(df)}")
    print(f"Saved to: {output_path}")

if __name__ == "__main__":
    main()