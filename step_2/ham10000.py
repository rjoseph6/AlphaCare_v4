import os
import shutil
import pandas as pd
from sklearn.model_selection import train_test_split

# Paths to the image directories
image_dirs = [
    '../datasets/ham10000_2/HAM10000_images_part_1',
    '../datasets/ham10000_2/HAM10000_images_part_2'
]  # Update these paths as needed

# Path to the HAM10000 metadata CSV file
metadata_path = '../datasets/ham10000_2/HAM10000_metadata.csv'  # Update this path as needed

# Destination directories
destination_path = '../datasets/cancer_vs_noncancer_v2'
train_dest_path = os.path.join(destination_path, 'train')
val_dest_path = os.path.join(destination_path, 'val')
test_dest_path = os.path.join(destination_path, 'test')  # Added test directory

# Ensure destination directories exist
for dest in [train_dest_path, val_dest_path, test_dest_path]:
    os.makedirs(os.path.join(dest, 'cancer'), exist_ok=True)
    os.makedirs(os.path.join(dest, 'non-cancerous'), exist_ok=True)

# Function to count images in a directory
def count_images_in_directory(directory):
    if not os.path.exists(directory):
        return 0
    return len([
        f for f in os.listdir(directory)
        if os.path.isfile(os.path.join(directory, f))
    ])

# Count existing images before adding HAM10000
pre_existing_counts = {
    'train_cancer': count_images_in_directory(os.path.join(train_dest_path, 'cancer')),
    'val_cancer': count_images_in_directory(os.path.join(val_dest_path, 'cancer')),
    'test_cancer': count_images_in_directory(os.path.join(test_dest_path, 'cancer')),
    'train_non_cancerous': count_images_in_directory(os.path.join(train_dest_path, 'non-cancerous')),
    'val_non_cancerous': count_images_in_directory(os.path.join(val_dest_path, 'non-cancerous')),
    'test_non_cancerous': count_images_in_directory(os.path.join(test_dest_path, 'non-cancerous'))
}

# Read the metadata CSV file
metadata = pd.read_csv(metadata_path)

# Define cancerous and non-cancerous labels based on 'dx' column
cancerous_labels = ['akiec', 'bcc', 'mel']  # Cancerous classes
non_cancerous_labels = ['bkl', 'df', 'nv', 'vasc']  # Non-cancerous classes

# Create a mapping from 'dx' to cancerous/non-cancerous labels
metadata['label'] = metadata['dx'].apply(lambda x: 'cancer' if x in cancerous_labels else 'non-cancerous')

# Initialize counters
image_paths = []
labels = []
found_count = 0
not_found_count = 0

# Collect image paths and labels
for idx, row in metadata.iterrows():
    image_id = row['image_id']
    label = row['label']
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

# Split into train_val and test sets first (90% train_val, 10% test)
train_val_paths, test_paths, train_val_labels, test_labels = train_test_split(
    image_paths, labels, test_size=0.1, stratify=labels, random_state=42
)

# Then split train_val into train and val sets (approx. 80% train, 10% val)
train_paths, val_paths, train_labels, val_labels = train_test_split(
    train_val_paths, train_val_labels, test_size=0.1111, stratify=train_val_labels, random_state=42
)

# Initialize counters for images added from HAM10000
added_counts = {
    'train_cancer': 0,
    'val_cancer': 0,
    'test_cancer': 0,
    'train_non_cancerous': 0,
    'val_non_cancerous': 0,
    'test_non_cancerous': 0
}

# Initialize a counter for images missed during copying
copy_missed_count = 0

# Function to copy images
def copy_images(paths, labels, dest_subdir, split_name):
    global copy_missed_count
    for img_path, label in zip(paths, labels):
        img_name = os.path.basename(img_path)
        # Prefix image name to avoid conflicts
        new_img_name = f"ham10000_{img_name}"
        dest_dir = os.path.join(dest_subdir, label)
        os.makedirs(dest_dir, exist_ok=True)
        dest_path = os.path.join(dest_dir, new_img_name)
        if not os.path.exists(dest_path):
            try:
                shutil.copy2(img_path, dest_path)
                # Update counters
                key = f"{split_name}_{label.replace('-', '_')}"
                if key in added_counts:
                    added_counts[key] += 1
            except Exception as e:
                print(f"Error copying {img_path} to {dest_path}: {e}")
                copy_missed_count += 1

# Copy training images
copy_images(train_paths, train_labels, train_dest_path, 'train')

# Copy validation images
copy_images(val_paths, val_labels, val_dest_path, 'val')

# Copy test images
copy_images(test_paths, test_labels, test_dest_path, 'test')

# Count images after adding HAM10000
post_existing_counts = {
    'train_cancer': count_images_in_directory(os.path.join(train_dest_path, 'cancer')),
    'val_cancer': count_images_in_directory(os.path.join(val_dest_path, 'cancer')),
    'test_cancer': count_images_in_directory(os.path.join(test_dest_path, 'cancer')),
    'train_non_cancerous': count_images_in_directory(os.path.join(train_dest_path, 'non-cancerous')),
    'val_non_cancerous': count_images_in_directory(os.path.join(val_dest_path, 'non-cancerous')),
    'test_non_cancerous': count_images_in_directory(os.path.join(test_dest_path, 'non-cancerous'))
}

# Calculate the number of images added
total_added = {
    'train_cancer': post_existing_counts['train_cancer'] - pre_existing_counts.get('train_cancer', 0),
    'val_cancer': post_existing_counts['val_cancer'] - pre_existing_counts.get('val_cancer', 0),
    'test_cancer': post_existing_counts['test_cancer'] - pre_existing_counts.get('test_cancer', 0),
    'train_non_cancerous': post_existing_counts['train_non_cancerous'] - pre_existing_counts.get('train_non_cancerous', 0),
    'val_non_cancerous': post_existing_counts['val_non_cancerous'] - pre_existing_counts.get('val_non_cancerous', 0),
    'test_non_cancerous': post_existing_counts['test_non_cancerous'] - pre_existing_counts.get('test_non_cancerous', 0),
}

# Print counts
print("\nCounts Before Adding HAM10000:")
print(f"  Train - Cancer: {pre_existing_counts.get('train_cancer', 0)}")
print(f"  Validation - Cancer: {pre_existing_counts.get('val_cancer', 0)}")
print(f"  Test - Cancer: {pre_existing_counts.get('test_cancer', 0)}")
print(f"  Train - Non-Cancerous: {pre_existing_counts.get('train_non_cancerous', 0)}")
print(f"  Validation - Non-Cancerous: {pre_existing_counts.get('val_non_cancerous', 0)}")
print(f"  Test - Non-Cancerous: {pre_existing_counts.get('test_non_cancerous', 0)}")

print("\nNumber of images added from HAM10000:")
print(f"---------------------- Training Set ----------------------")
print(f"  Train - Cancer: {total_added['train_cancer']}")
print(f"  Train - Non-Cancerous: {total_added['train_non_cancerous']}")
print(f"---------------------- Validation Set ----------------------")
print(f"  Validation - Cancer: {total_added['val_cancer']}")
print(f"  Validation - Non-Cancerous: {total_added['val_non_cancerous']}")
print(f"---------------------- Test Set ----------------------")
print(f"  Test - Cancer: {total_added['test_cancer']}")
print(f"  Test - Non-Cancerous: {total_added['test_non_cancerous']}")

print(f"\nTotal images missed during copying: {copy_missed_count}")