import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import cv2
import shutil
from sklearn.model_selection import train_test_split

# Constants
IMG_SIZE = 224

# Directories
data_dir = '../datasets/aptos2019/'
img_dir_train = os.path.join(data_dir, 'train_images')
img_dir_test = os.path.join(data_dir, 'test_images')

# Read CSV files
train_df = pd.read_csv(os.path.join(data_dir, 'train.csv'))
test_df = pd.read_csv(os.path.join(data_dir, 'test.csv'))

print(f"Train: {train_df.head()}\n")
print(f"Test: {test_df.head()}")

# Print shapes
print(f"Train shape: {train_df.shape}")
print(f"Test shape: {test_df.shape}")

# Plot class distribution
train_df['diagnosis'].value_counts().plot(kind='bar')
plt.title('Diagnosis')
plt.xlabel('Diagnosis')
plt.ylabel('Count')
plt.savefig('class_count_bar.png')
plt.close()

# Define the new dataset directory
new_data_dir = 'processed_data'

# Define train, validation, and test directories
train_dir = os.path.join(new_data_dir, 'train')
val_dir = os.path.join(new_data_dir, 'validation')
test_dir = os.path.join(new_data_dir, 'test')

# Get unique classes
classes = sorted(train_df['diagnosis'].unique())

# Create directory structure
for split_dir in [train_dir, val_dir, test_dir]:
    os.makedirs(split_dir, exist_ok=True)
    for cls in classes:
        cls_dir = os.path.join(split_dir, str(cls))
        os.makedirs(cls_dir, exist_ok=True)

# Split the data into train, temp (validation + test)
train_df, temp_df = train_test_split(
    train_df,
    test_size=0.20,  # 20% for temp (validation + test)
    random_state=42,
    stratify=train_df['diagnosis']
)

# Further split temp into validation and test
val_df, test_df_split = train_test_split(
    temp_df,
    test_size=0.50,  # Split temp equally into validation and test (10% each)
    random_state=42,
    stratify=temp_df['diagnosis']
)

print(f"Training set shape: {train_df.shape}")
print(f"Validation set shape: {val_df.shape}")
print(f"Test set shape: {test_df_split.shape}")

def copy_images(df, destination_dir):
    """
    Copies images to the destination directory organized by class.

    Parameters:
    - df (DataFrame): Pandas DataFrame containing 'id_code' and 'diagnosis' columns.
    - destination_dir (str): Path to the destination directory.
    """
    for idx, row in df.iterrows():
        img_id = row['id_code']
        label = row['diagnosis']
        src_path = os.path.join(img_dir_train, f"{img_id}.png")
        dst_path = os.path.join(destination_dir, str(label), f"{img_id}.png")
        
        if not os.path.exists(src_path):
            print(f"Source image not found: {src_path}")
            continue
        
        shutil.copy(src_path, dst_path)

# Copy training images
print("Copying training images...")
copy_images(train_df, train_dir)
print("Training images copied.")

# Copy validation images
print("Copying validation images...")
copy_images(val_df, val_dir)
print("Validation images copied.")

# Copy test images
print("Copying test images...")
copy_images(test_df_split, test_dir)
print("Test images copied.")

# Optional: Verify the folder structure
def verify_structure(base_dir, classes):
    for cls in classes:
        cls_path = os.path.join(base_dir, str(cls))
        if not os.path.exists(cls_path):
            print(f"Missing directory: {cls_path}")
        else:
            num_images = len(os.listdir(cls_path))
            print(f"Class {cls}: {num_images} images")

print("\nTraining Directory Structure:")
verify_structure(train_dir, classes)

print("\nValidation Directory Structure:")
verify_structure(val_dir, classes)

print("\nTest Directory Structure:")
verify_structure(test_dir, classes)

print("\nDataset organization complete!")