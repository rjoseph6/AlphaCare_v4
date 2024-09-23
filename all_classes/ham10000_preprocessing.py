import os
import shutil
import pandas as pd
from sklearn.model_selection import train_test_split

# ------------------------------
# Configuration
# ------------------------------

# Paths to the image directories
image_dirs = [
    '../datasets/ham10000_2/HAM10000_images_part_1',
    '../datasets/ham10000_2/HAM10000_images_part_2'
]  # Update these paths as needed

# Path to the HAM10000 metadata CSV file
metadata_path = '../datasets/ham10000_2/HAM10000_metadata.csv'  # Update this path as needed

# Destination directories
destination_path = '../datasets/ham10000_classified'  # New destination path for all classes
train_dest_path = os.path.join(destination_path, 'train')
val_dest_path = os.path.join(destination_path, 'val')
test_dest_path = os.path.join(destination_path, 'test')  # Added test directory

# List of all classes
classes = ['akiec', 'bcc', 'mel', 'bkl', 'df', 'nv', 'vasc']

# ------------------------------
# Create Destination Directories
# ------------------------------

def create_class_folders(base_path, classes):
    for cls in classes:
        cls_dir = os.path.join(base_path, cls)
        os.makedirs(cls_dir, exist_ok=True)

# Ensure destination directories exist with all class subdirectories
for dest in [train_dest_path, val_dest_path, test_dest_path]:
    create_class_folders(dest, classes)

# ------------------------------
# Helper Functions
# ------------------------------

# Function to count images in a directory
def count_images_in_directory(directory):
    if not os.path.exists(directory):
        return 0
    return len([
        f for f in os.listdir(directory)
        if os.path.isfile(os.path.join(directory, f)) and f.lower().endswith(('.jpg', '.jpeg', '.png'))
    ])

# Function to count images across all classes for a given split
def get_all_class_counts(base_path, classes):
    counts = {}
    for cls in classes:
        cls_dir = os.path.join(base_path, cls)
        counts[cls] = count_images_in_directory(cls_dir)
    return counts

# Function to copy images
def copy_images(paths, labels, dest_subdir, split_name):
    global copy_missed_count
    for img_path, label in zip(paths, labels):
        img_name = os.path.basename(img_path)
        # Prefix image name to avoid conflicts
        new_img_name = f"ham10000_{img_name}"
        dest_dir = os.path.join(dest_subdir, label)
        dest_path = os.path.join(dest_dir, new_img_name)
        if not os.path.exists(dest_path):
            try:
                shutil.copy2(img_path, dest_path)
                # Update counters
                key = f"{split_name}_{label}"
                if key in added_counts:
                    added_counts[key] += 1
            except Exception as e:
                print(f"Error copying {img_path} to {dest_path}: {e}")
                copy_missed_count += 1

# ------------------------------
# Initialize Counters
# ------------------------------

# Initialize counters for images added from HAM10000
added_counts = {f"{split}_{cls}": 0 for split in ['train', 'val', 'test'] for cls in classes}

# Initialize a counter for images missed during copying
copy_missed_count = 0

# ------------------------------
# Read and Process Metadata
# ------------------------------

# Read the metadata CSV file
metadata = pd.read_csv(metadata_path)

# Verify that all classes in 'dx' are in the defined classes list
unique_labels = metadata['dx'].unique()
for label in unique_labels:
    if label not in classes:
        raise ValueError(f"Label '{label}' in metadata is not defined in the classes list.")

# Collect image paths and labels
image_paths = []
labels = []
found_count = 0
not_found_count = 0

for idx, row in metadata.iterrows():
    image_id = row['image_id']
    label = row['dx']
    # Try to find the image file in the image directories with different extensions
    found = False
    for img_dir in image_dirs:
        for ext in ['.jpg', '.jpeg', '.png']:
            img_path = os.path.join(img_dir, image_id + ext)
            if os.path.exists(img_path):
                image_paths.append(img_path)
                labels.append(label)
                found = True
                found_count += 1
                break  # Exit the extension loop
        if found:
            break  # Exit the image_dirs loop
    if not found:
        not_found_count += 1

print(f"\nTotal images found: {found_count}")
print(f"Total images not found: {not_found_count}")

# ------------------------------
# Count Existing Images Before Adding HAM10000
# ------------------------------

pre_existing_counts = {
    split: get_all_class_counts(os.path.join(destination_path, split), classes)
    for split in ['train', 'val', 'test']
}

# ------------------------------
# Split the Data
# ------------------------------

# Split into train_val and test sets first (90% train_val, 10% test)
train_val_paths, test_paths, train_val_labels, test_labels = train_test_split(
    image_paths, labels, test_size=0.1, stratify=labels, random_state=42
)

# Then split train_val into train and val sets (approx. 80% train, 10% val)
# Since 0.1111 * 0.9 ≈ 0.1, this ensures approximately 80% train, 10% val, 10% test
train_paths, val_paths, train_labels, val_labels = train_test_split(
    train_val_paths, train_val_labels, test_size=0.1111, stratify=train_val_labels, random_state=42
)

# ------------------------------
# Copy Images to Destination Folders
# ------------------------------

# Copy training images
copy_images(train_paths, train_labels, train_dest_path, 'train')

# Copy validation images
copy_images(val_paths, val_labels, val_dest_path, 'val')

# Copy test images
copy_images(test_paths, test_labels, test_dest_path, 'test')

# ------------------------------
# Count Images After Adding HAM10000
# ------------------------------

post_existing_counts = {
    'train': get_all_class_counts(train_dest_path, classes),
    'val': get_all_class_counts(val_dest_path, classes),
    'test': get_all_class_counts(test_dest_path, classes)
}

# Calculate the number of images added
total_added = {
    split: {cls: post_existing_counts[split][cls] - pre_existing_counts[split][cls]
            for cls in classes}
    for split in ['train', 'val', 'test']
}

# ------------------------------
# Print Counts
# ------------------------------

# Print counts before adding HAM10000
print("\nCounts Before Adding HAM10000:")
for split in ['train', 'val', 'test']:
    for cls in classes:
        key = f"{split}_{cls}"
        print(f"  {split.capitalize()} - {cls}: {pre_existing_counts[split][cls]}")

# Print counts after adding HAM10000
print("\nNumber of images added from HAM10000:")
for split in ['train', 'val', 'test']:
    print(f"---------------------- {split.capitalize()} Set ----------------------")
    for cls in classes:
        print(f"  {split.capitalize()} - {cls}: {total_added[split].get(cls, 0)}")

print(f"\nTotal images missed during copying: {copy_missed_count}")