import warnings
warnings.filterwarnings("ignore", category=UserWarning)
import cv2
import mediapipe as mp
import numpy as np
import joblib

# --- Load the trained model and label encoder ---
model = joblib.load("models/letter_classifier.pkl")
label_encoder = joblib.load("models/letter_label_encoder.pkl")

# --- MediaPipe setup (video mode this time, not static image mode) ---
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,  # False = optimized for continuous video frames
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# --- Open the webcam ---
# 0 is usually the default/first camera. Since you have an external camera
# with no built-in one, 0 should be it - but we may need to adjust this.
CAMERA_INDEX = 0
cap = cv2.VideoCapture(CAMERA_INDEX)

if not cap.isOpened():
    print(f"ERROR: Could not open camera at index {CAMERA_INDEX}")
    exit()

def extract_normalized_landmarks(hand_landmarks):
    """
    Takes MediaPipe's hand landmarks for ONE hand and returns
    a normalized 63-length feature vector (same normalization as training).
    """
    coords = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark])

    # Wrist-relative (translation normalization)
    wrist = coords[0]
    coords = coords - wrist

    # Scale normalization (wrist to middle-finger-base distance)
    scale = np.linalg.norm(coords[9])
    if scale > 0:
        coords = coords / scale

    return coords.flatten()

print("\nStarting live letter recognition...")
print("Press 'q' to quit.\n")

while True:
    success, frame = cap.read()
    if not success:
        print("Failed to grab frame from camera.")
        break

    # Flip horizontally for a natural "mirror" view
    frame = cv2.flip(frame, 1)

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    # Build the same 126-length vector structure used in training
    landmarks_vector = np.zeros(126)
    predicted_letter = "No hand detected"

    if results.multi_hand_landmarks:
        for hand_index, hand_landmarks in enumerate(results.multi_hand_landmarks):
            if hand_index >= 2:
                break

            normalized = extract_normalized_landmarks(hand_landmarks)
            start = hand_index * 63
            landmarks_vector[start:start + 63] = normalized

            # Draw the hand skeleton on screen for visual feedback
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        # Predict using our trained model
        prediction = model.predict([landmarks_vector])
        predicted_letter = label_encoder.inverse_transform(prediction)[0]

    # Display the prediction on screen
    cv2.putText(
        frame, f"Letter: {predicted_letter}",
        (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2,
        (0, 255, 0), 3
    )

    cv2.imshow("ASL Letter Recognition - Live Test", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

