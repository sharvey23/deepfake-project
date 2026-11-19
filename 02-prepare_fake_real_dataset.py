
import json
import shutil
from pathlib import Path

# Project folders
base_path = Path("train_sample_videos")
output_path = Path("split_dataset")

# Assign each video to a dataset group.
# This is a small pipeline test, not a final evaluation.

video_splits = {
    # Training: 3 REAL + 3 FAKE
    "id1_0009.mp4": "train",
    "id3_0006.mp4": "train",
    "id9_0009.mp4": "train",
    "id0_id17_0006.mp4": "train",
    "id0_id28_0001.mp4": "train",
    "id1_id6_0001.mp4": "train",

    # Validation: 1 REAL + 1 FAKE
    "id16_0005.mp4": "val",
    "id0_id1_0001.mp4": "val",

    # Testing: 1 REAL + 1 FAKE
    "id21_0004.mp4": "test",
    "id1_id20_0003.mp4": "test",
}

# Load the real/fake labels
with open(base_path / "metadata.json", "r") as f:
    metadata = json.load(f)

# Check that all selected videos have labels
if set(metadata) != set(video_splits):
    raise ValueError(
        "Video filenames in metadata.json do not "
        "match the video_splits dictionary."
    )

# Create fresh output folders
if output_path.exists():
    shutil.rmtree(output_path)

for split in ["train", "val", "test"]:
    for label in ["real", "fake"]:
        (output_path / split / label).mkdir(
            parents=True, exist_ok=True
        )

# Copy cropped faces into their assigned groups
for filename, split in video_splits.items():
    label = metadata[filename]["label"].lower()

    if label not in ["real", "fake"]:
        raise ValueError(f"Invalid label for {filename}")

    video_name = Path(filename).stem
    faces_folder = base_path / video_name / "faces"
    destination = output_path / split / label

    if not faces_folder.exists():
        raise FileNotFoundError(
            f"Missing faces folder: {faces_folder}"
        )

    images = list(faces_folder.glob("*.png"))

    if not images:
        raise ValueError(
            f"No cropped faces found for {filename}"
        )

    for image in images:
        # Prefix filenames to prevent name collisions
        new_name = f"{video_name}_{image.name}"
        shutil.copy2(image, destination / new_name)

    print(
        f"{split.upper():5} | {label.upper():4} | "
        f"{filename} | {len(images)} faces"
    )

# Print dataset summary
print("\nDATASET SUMMARY")

for split in ["train", "val", "test"]:
    total = 0

    for label in ["real", "fake"]:
        folder = output_path / split / label
        count = len(list(folder.glob("*.png")))
        total += count
        print(f"{split}/{label}: {count}")

    print(f"{split} total: {total}\n")

print("Dataset preparation complete!")
