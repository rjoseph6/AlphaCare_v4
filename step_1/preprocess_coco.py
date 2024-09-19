import os
import random
import shutil
import requests
import zipfile
from pycocotools.coco import COCO
from tqdm import tqdm

# Set random seed for reproducibility
random.seed(42)

# Step 1: Download and Extract COCO Dataset with Progress Bars

def download_with_progress(url, save_path):
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))

    with open(save_path, 'wb') as file, tqdm(
        desc=f"Downloading {save_path}",
        total=total_size,
        unit='iB',
        unit_scale=True,
        unit_divisor=1024,
    ) as bar:
        for data in response.iter_content(chunk_size=1024):
            size = file.write(data)
            bar.update(size)

def extract_with_progress(zip_path, extract_path):
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        total_files = len(zip_ref.infolist())
        with tqdm(total=total_files, desc=f'Extracting {zip_path}') as pbar:
            for file in zip_ref.infolist():
                zip_ref.extract(member=file, path=extract_path)
                pbar.update(1)

# Create directories
coco_dir = 'coco_data'
os.makedirs(coco_dir, exist_ok=True)

# URLs for COCO 2017 dataset (train images and annotations)
train_images_url = 'http://images.cocodataset.org/zips/train2017.zip'
annotations_url = 'http://images.cocodataset.org/annotations/annotations_trainval2017.zip'

# Download and extract datasets
train_zip_path = os.path.join(coco_dir, 'train2017.zip')
annotations_zip_path = os.path.join(coco_dir, 'annotations_trainval2017.zip')

# Download train images
if not os.path.exists(train_zip_path):
    download_with_progress(train_images_url, train_zip_path)
else:
    print(f'{train_zip_path} already exists.')

# Extract train images
if not os.path.exists(os.path.join(coco_dir, 'train2017')):
    extract_with_progress(train_zip_path, coco_dir)
else:
    print('Train images already extracted.')

# Download annotations
if not os.path.exists(annotations_zip_path):
    download_with_progress(annotations_url, annotations_zip_path)
else:
    print(f'{annotations_zip_path} already exists.')

# Extract annotations
if not os.path.exists(os.path.join(coco_dir, 'annotations')):
    extract_with_progress(annotations_zip_path, coco_dir)
else:
    print('Annotations already extracted.')

# Step 2: Initialize COCO API

annotations_file = os.path.join(coco_dir, 'annotations', 'instances_train2017.json')
coco = COCO(annotations_file)

# Step 3: Prepare Data Directories

# Paths to COCO images
images_dir = os.path.join(coco_dir, 'train2017')

# Destination directories for non-skin images
train_dest_dir = 'datasets/train/non-skin'
val_dest_dir = 'datasets/val/non-skin'

os.makedirs(train_dest_dir, exist_ok=True)
os.makedirs(val_dest_dir, exist_ok=True)

# Step 4: Select and Copy Images

# Get all image IDs
all_img_ids = coco.getImgIds()

# Shuffle and split image IDs into training and validation sets
random.shuffle(all_img_ids)
split_index = int(len(all_img_ids) * 0.8)
train_img_ids = all_img_ids[:split_index]
val_img_ids = all_img_ids[split_index:]

# Limit the number of images to balance with 'skin' images
num_train_images = 5000  # Adjust based on your 'skin' dataset size
num_val_images = 1000

train_img_ids = train_img_ids[:num_train_images]
val_img_ids = val_img_ids[:num_val_images]

# Function to copy images
def copy_images(img_ids, dest_dir):
    for img_id in img_ids:
        img_info = coco.loadImgs(img_id)[0]
        src_path = os.path.join(images_dir, img_info['file_name'])
        dest_path = os.path.join(dest_dir, img_info['file_name'])
        shutil.copyfile(src_path, dest_path)

# Copy images to destination directories
print('Copying training non-skin images...')
copy_images(train_img_ids, train_dest_dir)

print('Copying validation non-skin images...')
copy_images(val_img_ids, val_dest_dir)

print('COCO dataset preparation is complete.')