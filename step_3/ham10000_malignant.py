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
destination_path = '../datasets/malignant_dataset'
train_dest_path = os.path.join(destination_path, 'train')
val_dest_path = os.path.join(destination_path, 'val')
test_dest_path = os.path.join(destination_path, 'test')  # Added test directory

# Ensure destination directories exist and create subdirectories for each class
cancerous_classes = ['akiec', 'bcc', 'mel']  # Cancerous classes
for dest in [train_dest_path, val_dest_path, test_dest_path]:
    os.makedirs(dest, exist_ok=True)
    for cls in cancerous_classes:
        os.makedirs(os.path.join(dest, cls), exist_ok=True)

# Function to count images in a directory
def count_images_in_directory(directory):
    if not os.path.exists(directory):
        return 0
    return len([
        f for f in os.listdir(directory)
        if os.path.isfile(os.path.join(directory, f))
    ])

# Read the metadata CSV file
metadata = pd.read_csv(metadata_path)

# Filter metadata to include only the cancerous classes
metadata = metadata[metadata['dx'].isin(cancerous_classes)].reset_index(drop=True)

# Use 'dx' as the label since we're only dealing with the three cancerous classes
metadata['label'] = metadata['dx']

# Initialize counters
image_paths = []
labels = []
found_count = 0
not_found_count = 0

# Collect image paths and labels
for idx, row in metadata.iterrows():
    image_id = row['image_id']
    label = row['label']  # This will be 'akiec', 'bcc', or 'mel'
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

# Function to copy images
def copy_images(paths, labels, dest_subdir):
    for img_path, label in zip(paths, labels):
        img_name = os.path.basename(img_path)
        # Prefix image name to avoid conflicts
        new_img_name = f"ham10000_{img_name}"
        dest_dir = os.path.join(dest_subdir, label)  # Save in class-specific directory
        os.makedirs(dest_dir, exist_ok=True)
        dest_path = os.path.join(dest_dir, new_img_name)
        if not os.path.exists(dest_path):
            try:
                shutil.copy2(img_path, dest_path)
            except Exception as e:
                print(f"Error copying {img_path} to {dest_path}: {e}")

# Copy training images
copy_images(train_paths, train_labels, train_dest_path)

# Copy validation images
copy_images(val_paths, val_labels, val_dest_path)

# Copy test images
copy_images(test_paths, test_labels, test_dest_path)

print("\nDataset creation completed.")

def count_images_per_class(directory, classes):
    counts = {}
    for cls in classes:
        cls_dir = os.path.join(directory, cls)
        counts[cls] = count_images_in_directory(cls_dir)
    return counts

print("\nImage counts per class in each dataset split:")
print("Training set:", count_images_per_class(train_dest_path, cancerous_classes))
print("Validation set:", count_images_per_class(val_dest_path, cancerous_classes))
print("Test set:", count_images_per_class(test_dest_path, cancerous_classes))