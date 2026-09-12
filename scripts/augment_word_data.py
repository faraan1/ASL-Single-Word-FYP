import numpy as np
import os

# --- Configuration ---
DATA_DIR = "data/words/processed"
TARGET_PER_WORD = 100

# Load our extracted real sequences
X = np.load(os.path.join(DATA_DIR, "word_sequences.npy"))
y = np.load(os.path.join(DATA_DIR, "word_labels.npy"))

print(f"Loaded {len(X)} real sequences across {len(np.unique(y))} words")
print(f"Sequence shape: {X.shape}")

def rotate_sequence(sequence, max_angle=15):
    """Rotates all landmarks in the sequence slightly around the Z axis."""
    angle = np.radians(np.random.uniform(-max_angle, max_angle))
    cos_a, sin_a = np.cos(angle), np.sin(angle)
    rotation_matrix = np.array([[cos_a, -sin_a], [sin_a, cos_a]])

    augmented = sequence.copy()
    for frame_idx in range(len(augmented)):
        for hand_idx in range(2):  # 2 hands
            start = hand_idx * 63
            for point_idx in range(21):  # 21 landmarks per hand
                p_start = start + point_idx * 3
                xy = augmented[frame_idx, p_start:p_start + 2]
                augmented[frame_idx, p_start:p_start + 2] = rotation_matrix @ xy
    return augmented


def scale_sequence(sequence, scale_range=(0.9, 1.1)):
    """Slightly scales all coordinates up or down."""
    scale = np.random.uniform(*scale_range)
    return sequence * scale


def add_noise(sequence, noise_level=0.02):
    """Adds small random jitter to simulate tracking imprecision."""
    noise = np.random.normal(0, noise_level, sequence.shape)
    return sequence + noise


def time_warp(sequence, warp_factor_range=(0.85, 1.15)):
    """Slightly speeds up or slows down the motion by resampling."""
    factor = np.random.uniform(*warp_factor_range)
    original_length = len(sequence)
    new_length = int(original_length * factor)
    new_length = max(5, new_length)  # safety minimum

    indices = np.linspace(0, original_length - 1, new_length)
    warped = sequence[np.round(indices).astype(int)]

    # Resample back to the original fixed length
    final_indices = np.linspace(0, new_length - 1, original_length)
    return warped[np.round(final_indices).astype(int)]


def mirror_sequence(sequence):
    """Flips the sequence left-right (simulates opposite-handed signing)."""
    augmented = sequence.copy()
    # Flip the x-coordinate of every landmark (x is every 3rd value starting at 0)
    for hand_idx in range(2):
        start = hand_idx * 63
        for point_idx in range(21):
            x_index = start + point_idx * 3
            augmented[:, x_index] = -augmented[:, x_index]
    return augmented

def augment_sample(sequence):
    """Applies a random combination of augmentations to one sequence."""
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
    final_X = []
    final_y = []

    for word in unique_words:
        word_indices = np.where(y == word)[0]
        real_sequences = X[word_indices]
        real_count = len(real_sequences)

        # Keep all real samples
        for seq in real_sequences:
            final_X.append(seq)
            final_y.append(word)

        # Generate augmented samples until we hit our target
        needed = TARGET_PER_WORD - real_count
        for _ in range(needed):
            source_seq = real_sequences[np.random.randint(real_count)]
            augmented_seq = augment_sample(source_seq)
            final_X.append(augmented_seq)
            final_y.append(word)

        print(f"Word '{word}': {real_count} real + {needed} augmented = {real_count + needed} total")

    final_X = np.array(final_X)
    final_y = np.array(final_y)

    np.save(os.path.join(DATA_DIR, "word_sequences_augmented.npy"), final_X)
    np.save(os.path.join(DATA_DIR, "word_labels_augmented.npy"), final_y)

    print(f"\nTotal samples after augmentation: {len(final_X)}")
    print(f"Saved to word_sequences_augmented.npy and word_labels_augmented.npy")


if __name__ == "__main__":
    main()

