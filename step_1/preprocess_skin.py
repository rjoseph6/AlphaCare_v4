import os
import shutil
import random

def copy_and_split_images(source_dirs, train_dir, val_dir, val_split=0.1):
    # Create train and val directories if they don't exist
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)
    
    # Collect all images from the source directories
    all_images = []
    for source_dir in source_dirs:
        images = [f for f in os.listdir(source_dir) if os.path.isfile(os.path.join(source_dir, f))]
        all_images.extend([(source_dir, image) for image in images])
    
    # Shuffle images for randomness
    random.shuffle(all_images)
    
    # Split images into training and validation sets
    split_idx = int(len(all_images) * (1 - val_split))
    train_images = all_images[:split_idx]
    val_images = all_images[split_idx:]
    
    # Copy images to the train directory
    for source_dir, image in train_images:
        src = os.path.join(source_dir, image)
        dst = os.path.join(train_dir, image)
        shutil.copy(src, dst)
    
    # Copy images to the val directory
    for source_dir, image in val_images:
        src = os.path.join(source_dir, image)
        dst = os.path.join(val_dir, image)
        shutil.copy(src, dst)
    
    print(f"Images have been split: {len(train_images)} in {train_dir}, {len(val_images)} in {val_dir}.")

# Example usage
ham_dirs = ['../datasets/ham10000/HAM10000_images_part_1', '../datasets/ham10000/HAM10000_images_part_2']
train_directory = '../datasets/skin_nonskin/train/'
val_directory = '../datasets/skin_nonskin/val/'

copy_and_split_images(ham_dirs, train_directory, val_directory)



