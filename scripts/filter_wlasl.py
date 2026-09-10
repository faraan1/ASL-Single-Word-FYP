import json
import os

# --- Configuration ---
WLASL_JSON_PATH = "data/words/WLASL_v0.3.json"
VIDEOS_FOLDER = "data/words/videos"
OUTPUT_PATH = "data/words/filtered_words.json"

# Our finalized 22-word list
TARGET_WORDS = [
    "i", "you", "we", "they",
    "want", "drink", "water", "eat", "like",
    "play", "go", "come", "help", "stop",
    "outside", "home", "school",
    "where", "who",
    "yes", "no", "please", "thank you", "sorry"
]

def main():
    # Step 1: Load the full WLASL metadata
    with open(WLASL_JSON_PATH, "r") as f:
        wlasl_data = json.load(f)

    print(f"Total words in WLASL dataset: {len(wlasl_data)}")

    filtered = {}
    found_words = []
    missing_words = []

    # Step 2: Go through every entry, check if it's one of our target words
    for entry in wlasl_data:
        gloss = entry["gloss"].lower().strip()
        if gloss in TARGET_WORDS:
            # Step 3: For each matching word, check which video files actually exist on disk
            valid_instances = []
            for instance in entry["instances"]:
                video_id = instance["video_id"]
                video_path = os.path.join(VIDEOS_FOLDER, f"{video_id}.mp4")
                if os.path.exists(video_path):
                    valid_instances.append(instance)

            if valid_instances:
                filtered[gloss] = valid_instances
                found_words.append(gloss)

    # Step 4: Check which of our target words were NOT found at all
    for word in TARGET_WORDS:
        if word not in found_words:
            missing_words.append(word)

    # Step 5: Report results
    print(f"\nFound {len(found_words)} / {len(TARGET_WORDS)} target words:")
    for word in found_words:
        print(f"  - '{word}': {len(filtered[word])} video(s)")

    if missing_words:
        print(f"\nMISSING words (not found in WLASL): {missing_words}")

    # Step 6: Save the filtered result
    with open(OUTPUT_PATH, "w") as f:
        json.dump(filtered, f, indent=2)

    print(f"\nSaved filtered data to {OUTPUT_PATH}")

if __name__ == "__main__":
    main()