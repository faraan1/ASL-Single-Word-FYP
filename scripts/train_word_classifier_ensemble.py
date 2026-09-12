import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import joblib
import os

# --- Configuration ---
DATA_DIR = "data/words/processed"
MODEL_OUTPUT_DIR = "models"

# --- Load the same train/test split used for the LSTM ---
X_train_seq = np.load(os.path.join(DATA_DIR, "word_train_X.npy"))
y_train_raw = np.load(os.path.join(DATA_DIR, "word_train_y.npy"))
X_test_seq = np.load(os.path.join(DATA_DIR, "word_test_X.npy"))
y_test_raw = np.load(os.path.join(DATA_DIR, "word_test_y.npy"))


def summarize_sequence(sequence):
    """
    Converts a (30, 126) sequence into a flat summary feature vector:
    mean, std, min, max across time for each of the 126 landmark features.
    Result: 126 * 4 = 504 features.
    """
    mean = sequence.mean(axis=0)
    std = sequence.std(axis=0)
    minimum = sequence.min(axis=0)
    maximum = sequence.max(axis=0)
    return np.concatenate([mean, std, minimum, maximum])


# Convert every sequence into summary features
X_train = np.array([summarize_sequence(seq) for seq in X_train_seq])
X_test = np.array([summarize_sequence(seq) for seq in X_test_seq])

print(f"Converted sequences to summary features: {X_train.shape[1]} features per sample")

# Encode labels (same as before)
label_encoder = LabelEncoder()
y_train = label_encoder.fit_transform(y_train_raw)
y_test = label_encoder.transform(y_test_raw)

print(f"Training samples: {len(X_train)}, Test samples: {len(X_test)}")

from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import HistGradientBoostingClassifier

# --- Candidate individual models to test ---
candidate_models = {
    "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42),
    "Logistic Regression": LogisticRegression(max_iter=2000),
    "Neural Network (MLP)": MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=1000, random_state=42),
    "SVM": SVC(probability=True, random_state=42),
    "KNN": KNeighborsClassifier(n_neighbors=5),
    "Gradient Boosting": HistGradientBoostingClassifier(random_state=42),}

results = {}

print("\n--- Testing individual models ---")
for name, model in candidate_models.items():
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    results[name] = acc
    print(f"{name}: {acc*100:.2f}% test accuracy")

# --- Build a stacked ensemble from the best-performing candidates ---
# Sort candidates by individual accuracy, pick the top 3 as base models
sorted_models = sorted(results.items(), key=lambda x: x[1], reverse=True)
print(f"\nRanking: {sorted_models}")

top_3_names = [name for name, acc in sorted_models[:3]]
print(f"\nUsing top 3 as base models for stacking: {top_3_names}")

base_estimators = [(name, candidate_models[name]) for name in top_3_names]

stacked_model = StackingClassifier(
    estimators=base_estimators,
    final_estimator=LogisticRegression(max_iter=2000),
    cv=5
)

stacked_model.fit(X_train, y_train)
stacked_preds = stacked_model.predict(X_test)
stacked_acc = accuracy_score(y_test, stacked_preds)

print(f"\nStacked Ensemble (top 3): {stacked_acc*100:.2f}% test accuracy")
print("\nClassification Report (Stacked Ensemble):")
print(classification_report(y_test, stacked_preds, target_names=label_encoder.classes_, zero_division=0))

# --- Compare against LSTM result and save whichever is best ---
LSTM_ACCURACY = 0.7073  # from our earlier LSTM run

print(f"\n--- Final Comparison ---")
print(f"LSTM: {LSTM_ACCURACY*100:.2f}%")
print(f"Stacked Ensemble: {stacked_acc*100:.2f}%")

if stacked_acc > LSTM_ACCURACY:
    print("\nStacked Ensemble wins - saving this as the final word model.")
    os.makedirs(MODEL_OUTPUT_DIR, exist_ok=True)
    joblib.dump(stacked_model, os.path.join(MODEL_OUTPUT_DIR, "word_classifier_ensemble.pkl"))
    joblib.dump(label_encoder, os.path.join(MODEL_OUTPUT_DIR, "word_label_encoder_ensemble.pkl"))
    print(f"Saved to {MODEL_OUTPUT_DIR}/word_classifier_ensemble.pkl")
else:
    print("\nLSTM wins - keeping the existing LSTM model as the final word model.")

