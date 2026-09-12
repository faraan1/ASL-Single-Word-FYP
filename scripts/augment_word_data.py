import numpy as np
import os

# --- Configuration ---
DATA_DIR = "data/words/processed"
TARGET_PER_WORD = 100
RANDOM_SEED = 42

np.random.seed(RANDOM_SEED)

# Load our extracted real sequences
X = np.load(os.path.join(DATA_DIR, "word_sequences.npy"))
y = np.load(os.path.join(DATA_DIR, "word_labels.npy"))

print(f"Loaded {len(X)} real sequences across {len(np.unique(y))} words")


# --- Augmentation functions (unchanged from before) ---
def rotate_sequence(sequence, max_angle=15):
    angle = np.radians(np.random.uniform(-max_angle, max_angle))
    cos_a, sin_a = np.cos(angle), np.sin(angle)
    rotation_matrix = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
    augmented = sequence.copy()
    for frame_idx in range(len(augmented)):
        for hand_idx in range(2):
            start = hand_idx * 63
            for point_idx in range(21):
                p_start = start + point_idx * 3
                xy = augmented[frame_idx, p_start:p_start + 2]
                augmented[frame_idx, p_start:p_start + 2] = rotation_matrix @ xy
    return augmented


def scale_sequence(sequence, scale_range=(0.9, 1.1)):
    scale = np.random.uniform(*scale_range)
    return sequence * scale


def add_noise(sequence, noise_level=0.02):
    noise = np.random.normal(0, noise_level, sequence.shape)
    return sequence + noise


def time_warp(sequence, warp_factor_range=(0.85, 1.15)):
    factor = np.random.uniform(*warp_factor_range)
    original_length = len(sequence)
    new_length = max(5, int(original_length * factor))
    indices = np.linspace(0, original_length - 1, new_length)
    warped = sequence[np.round(indices).astype(int)]
    final_indices = np.linspace(0, new_length - 1, original_length)
    return warped[np.round(final_indices).astype(int)]


def mirror_sequence(sequence):
    augmented = sequence.copy()
    for hand_idx in range(2):
        start = hand_idx * 63
        for point_idx in range(21):
            x_index = start + point_idx * 3
            augmented[:, x_index] = -augmented[:, x_index]
    return augmented


def augment_sample(sequence):
    augmented = sequence.copy()
    if np.random.rand() < 0.7:
        augmented = rotate_sequence(augmented)
    if np.random.rand() < 0.7:
        augmented = scale_sequence(augmented)
    if np.random.rand() < 0.8:
        augmented = add_noise(augmented)
    if np.random.rand() < 0.5:
        augmented = time_warp(augmented)
    if np.random.rand() < 0.3:
        augmented = mirror_sequence(augmented)
    return augmented


def main():
    unique_words = np.unique(y)

    train_X, train_y = [], []
    test_X, test_y = [], []

    for word in unique_words:
        word_indices = np.where(y == word)[0]
        real_sequences = X[word_indices]
        real_count = len(real_sequences)

        # Shuffle this word's real samples before splitting
        shuffled = real_sequences[np.random.permutation(real_count)]

        # Reserve at least 1 real sample for testing (never augmented)
        test_count = max(1, round(real_count * 0.2))
        test_samples = shuffled[:test_count]
        train_samples = shuffled[test_count:]

        # Test set: pure real data, untouched
        for seq in test_samples:
            test_X.append(seq)
            test_y.append(word)

        # Train set: real samples + augmented copies, built ONLY from train_samples
        for seq in train_samples:
            train_X.append(seq)
            train_y.append(word)

        needed = TARGET_PER_WORD - len(train_samples)
        for _ in range(needed):
            source_seq = train_samples[np.random.randint(len(train_samples))]
            augmented_seq = augment_sample(source_seq)
            train_X.append(augmented_seq)
            train_y.append(word)

        print(f"Word '{word}': {real_count} real -> {len(train_samples)} train real "
              f"+ {needed} augmented = {len(train_samples) + needed} train total, "
              f"{test_count} test (real, untouched)")

    train_X, train_y = np.array(train_X), np.array(train_y)
    test_X, test_y = np.array(test_X), np.array(test_y)

    np.save(os.path.join(DATA_DIR, "word_train_X.npy"), train_X)
    np.save(os.path.join(DATA_DIR, "word_train_y.npy"), train_y)
    np.save(os.path.join(DATA_DIR, "word_test_X.npy"), test_X)
    np.save(os.path.join(DATA_DIR, "word_test_y.npy"), test_y)

    print(f"\nTrain set: {len(train_X)} samples")
    print(f"Test set: {len(test_X)} samples (100% real, never augmented)")
    print("Saved word_train_X.npy, word_train_y.npy, word_test_X.npy, word_test_y.npy")


if __name__ == "__main__":
    main() 