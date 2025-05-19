import os
import shutil
import random

# Set paths
dataset_dir = '.'  # Root folder containing images/ and labels/
images_dir = os.path.join(dataset_dir, 'images')
labels_dir = os.path.join(dataset_dir, 'labels')

# Output folders
splits = ['train', 'val', 'test']

# Create output directories if they don't exist
for split in splits:
    os.makedirs(os.path.join(images_dir, split), exist_ok=True)
    os.makedirs(os.path.join(labels_dir, split), exist_ok=True)

# Split ratios (add up to 1.0)
train_ratio = 0.7
val_ratio = 0.2
test_ratio = 0.1

assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "Ratios must add to 1.0"

# Get all image files in the root images_dir (not in split folders)
all_images = [
    f for f in os.listdir(images_dir)
    if os.path.isfile(os.path.join(images_dir, f))
    and (f.endswith('.jpg') or f.endswith('.png'))
]

# Shuffle files for randomness
random.seed(42)  # For reproducibility
random.shuffle(all_images)

total = len(all_images)
train_end = int(total * train_ratio)
val_end = train_end + int(total * val_ratio)

train_files = all_images[:train_end]
val_files = all_images[train_end:val_end]
test_files = all_images[val_end:]

def move_files(file_list, split):
    for file_name in file_list:
        # Move image
        src_img = os.path.join(images_dir, file_name)
        dst_img = os.path.join(images_dir, split, file_name)
        if os.path.exists(src_img):
            shutil.move(src_img, dst_img)

        # Move label (same name, .txt)
        label_name = os.path.splitext(file_name)[0] + '.txt'
        src_label = os.path.join(labels_dir, label_name)
        dst_label = os.path.join(labels_dir, split, label_name)
        if os.path.exists(src_label):
            shutil.move(src_label, dst_label)
        else:
            print(f"Warning: Label file not found for {file_name}")

# Move files to respective folders
move_files(train_files, 'train')
move_files(val_files, 'val')
move_files(test_files, 'test')

print(f"Dataset split complete:\nTrain: {len(train_files)}\nVal: {len(val_files)}\nTest: {len(test_files)}")