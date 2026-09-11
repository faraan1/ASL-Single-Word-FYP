import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import os

# --- Load the data ---
DATA_PATH = "data/letters/processed/letter_landmarks.csv"
MODEL_OUTPUT_DIR = "models"

df = pd.read_csv(DATA_PATH)

# Separate features (the 126 landmark numbers) from labels (the letter)
X = df.drop(columns=["label"])
y_raw = df["label"]

# Convert letter labels (A, B, C...) into numbers (0, 1, 2...) - models need numeric labels
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y_raw)

print(f"Loaded {len(X)} samples across {len(label_encoder.classes_)} classes")
print(f"Classes: {list(label_encoder.classes_)}")

# --- Split into train and test sets ---
# 80% for training, 20% held back to test how well the model generalizes
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\nTraining samples: {len(X_train)}")
print(f"Testing samples: {len(X_test)}")

# --- Build the stacked ensemble ---
# Base models: Neural Network + Random Forest
base_models = [
    ("neural_network", MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=500, random_state=42)),
    ("random_forest", RandomForestClassifier(n_estimators=200, random_state=42))
]

# Meta-learner: Logistic Regression combines the base models' predictions
stacked_model = StackingClassifier(
    estimators=base_models,
    final_estimator=LogisticRegression(max_iter=1000),
    cv=5
)

# --- Train the model ---
print("\nTraining stacked ensemble... (this may take a few minutes)")
stacked_model.fit(X_train, y_train)

# --- Evaluate on the held-out test set ---
y_pred = stacked_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"\nTest Accuracy: {accuracy * 100:.2f}%")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))

# --- Save the trained model and label encoder ---
os.makedirs(MODEL_OUTPUT_DIR, exist_ok=True)
joblib.dump(stacked_model, os.path.join(MODEL_OUTPUT_DIR, "letter_classifier.pkl"))
joblib.dump(label_encoder, os.path.join(MODEL_OUTPUT_DIR, "letter_label_encoder.pkl"))

print(f"\nModel saved to {MODEL_OUTPUT_DIR}/letter_classifier.pkl")
print(f"Label encoder saved to {MODEL_OUTPUT_DIR}/letter_label_encoder.pkl")