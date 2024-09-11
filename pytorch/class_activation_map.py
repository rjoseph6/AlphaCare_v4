import numpy as np
import pandas as pd
import os
import torch
from glob import glob
from PIL import Image
from sklearn.model_selection import train_test_split
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt

# Check if GPU is available, otherwise fall back to CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

print("Hello")

base_skin_dir = '../mnist/archive/'

# Merging images from both folders HAM10000_images_part1.zip and HAM10000_images_part2.zip into one dictionary
imageid_path_dict = {os.path.splitext(os.path.basename(x))[0]: x
                     for x in glob(os.path.join(base_skin_dir, '*', '*.jpg'))}

# This dictionary is useful for displaying more human-friendly labels later on
lesion_type_dict = {
    'nv': 'Melanocytic nevi',
    'mel': 'Melanoma',
    'bkl': 'Benign keratosis-like lesions',
    'bcc': 'Basal cell carcinoma',
    'akiec': 'Actinic keratoses',
    'vasc': 'Vascular lesions',
    'df': 'Dermatofibroma'
}
skin_df = pd.read_csv(os.path.join(base_skin_dir, 'HAM10000_metadata.csv'))
# Creating New Columns for better readability
skin_df['path'] = skin_df['image_id'].map(imageid_path_dict.get)
skin_df['cell_type'] = skin_df['dx'].map(lesion_type_dict.get) 
skin_df['cell_type_idx'] = pd.Categorical(skin_df['cell_type']).codes
# Shrink dataset
print(f"Original Size: {skin_df.shape}")
skin_df = skin_df.sample(frac=0.01) # shuffle the dataset
print(f"Shrunk Size: {skin_df.shape}")
# Fill missing values in 'age'
skin_df['age'].fillna(skin_df['age'].mean(), inplace=True)
# Resize images and convert to arrays
skin_df['image'] = skin_df['path'].map(lambda x: np.asarray(Image.open(x).resize((100, 75))))
# Features and target
features = skin_df.drop(columns=['cell_type_idx'], axis=1)
target = skin_df['cell_type_idx']
# Train-test split
x_train_o, x_test_o, y_train_o, y_test_o = train_test_split(features, target, test_size=0.20, random_state=1234)
# Convert images to numpy arrays
x_train = np.asarray(x_train_o['image'].tolist())
x_test = np.asarray(x_test_o['image'].tolist())
# Normalize the images
x_train_mean = np.mean(x_train)
x_train_std = np.std(x_train)
x_test_mean = np.mean(x_test)
x_test_std = np.std(x_test)
x_train = (x_train - x_train_mean) / x_train_std
x_test = (x_test - x_test_mean) / x_test_std
# Ensure y_train_o and y_test_o are integers for one-hot encoding
y_train_o = y_train_o.astype(int)
y_test_o = y_test_o.astype(int)
# Perform one-hot encoding on the labels using PyTorch's F.one_hot
y_train = F.one_hot(torch.tensor(y_train_o.values), num_classes=7).float()
y_test = F.one_hot(torch.tensor(y_test_o.values), num_classes=7).float()
# Split train into train and validation sets
x_train, x_validate, y_train, y_validate = train_test_split(x_train, y_train, test_size=0.1, random_state=2)
# Debug: Print shapes of datasets
print(f"x_train shape before reshaping: {x_train.shape}")
print(f"x_test shape before reshaping: {x_test.shape}")
# Reshape images for PyTorch (channels-first format)
x_train = torch.tensor(x_train).permute(0, 3, 1, 2).float().to(device)  # (batch_size, channels, height, width)
x_test = torch.tensor(x_test).permute(0, 3, 1, 2).float().to(device)    # (batch_size, channels, height, width)
x_validate = torch.tensor(x_validate).permute(0, 3, 1, 2).float().to(device)  # (batch_size, channels, height, width)
y_train = y_train.to(device)
y_test = y_test.to(device)
y_validate = y_validate.to(device)
print(f"x_train shape after reshaping: {x_train.shape}")
print(f"x_test shape after reshaping: {x_test.shape}")
print(f"x_validate shape after reshaping: {x_validate.shape}")
input_shape = (3, 75, 100)  # PyTorch expects (channels, height, width)
num_classes = 7

class SimpleCNN(nn.Module):
    def __init__(self, num_classes=7):
        super(SimpleCNN, self).__init__()

        # First convolutional layer
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.dropout1 = nn.Dropout(0.5)  # Apply dropout after first conv layer
        # Second convolutional layer
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.dropout2 = nn.Dropout(0.5)  # Apply dropout after second conv layer
        # Third convolutional layer
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.dropout3 = nn.Dropout(0.5)  # Apply dropout after third conv layer
        self.pool = nn.MaxPool2d(2, 2)  # 2x2 Max Pooling
        # Fully connected layers
        self.fc1 = nn.Linear(128 * 9 * 12, 512)  # Flatten after pooling
        self.dropout_fc1 = nn.Dropout(0.5)  # Apply dropout after first fully connected layer
        self.fc2 = nn.Linear(512, num_classes)  # Output layer
    def forward(self, x):
        # First conv layer + ReLU + Dropout + Pooling
        x = self.pool(F.relu(self.conv1(x)))
        x = self.dropout1(x)
        # Second conv layer + ReLU + Dropout + Pooling
        x = self.pool(F.relu(self.conv2(x)))
        x = self.dropout2(x)
        # Third conv layer + ReLU + Dropout + Pooling
        x = self.pool(F.relu(self.conv3(x)))
        x = self.dropout3(x)
        # Flatten the tensor for fully connected layers
        x = x.view(-1, 128 * 9 * 12)
        # Fully connected layer + ReLU + Dropout
        x = F.relu(self.fc1(x))
        x = self.dropout_fc1(x)
        # Output layer
        x = self.fc2(x)
        return x

import numpy as np
import matplotlib.pyplot as plt
import cv2

# Initialize the model architecture
model = SimpleCNN(num_classes=num_classes).to(device)
# Load the state_dict into the model
model.load_state_dict(torch.load('../weights/model_v1_acc_73.pth', weights_only=True))
# Set the model to evaluation mode
model.eval()
print("Model loaded successfully!")

print("\n----------------------- Heatmap Generation -----------------------\n")

# Perform Monte Carlo Dropout Inference on one test sample (e.g., the first sample)
test_sample = x_validate[0].unsqueeze(0)  # Select the first test sample and add a batch dimension
print(f"Test sample shape: {test_sample.shape}")

class SaveFeatures():
    features = None
    def __init__(self, m): self.hook = m.register_forward_hook(self.hook_fn)
    def hook_fn(self, module, input, output): self.features = ((output.cpu()).data).numpy()
    def remove(self): self.hook.remove()

# Hook the final convolutional layer (conv3 in this case)
final_layer = model.conv3

# SaveFeatures class to hook into the layer
activated_features = SaveFeatures(final_layer)

# Perform forward pass on the test sample
prediction = model(test_sample)
pred_probabilities = F.softmax(prediction, dim=1).data.squeeze()
activated_features.remove()

# Get the top class index
_, top_class_idx = torch.topk(pred_probabilities, 1)

# Generate Class Activation Map (CAM)
def getCAM(feature_conv, weight_fc, class_idx, target_size, threshold=0.6):
    _, nc, h, w = feature_conv.shape  # nc: number of channels, h: height, w: width
    cam = np.zeros((h, w), dtype=np.float32)

    # Multiply each channel by the corresponding weight, limit to available channels
    for i in range(min(nc, len(weight_fc[class_idx]))):
        cam += weight_fc[class_idx][i] * feature_conv[0, i, :, :]
    
    # Normalize the CAM
    cam = cam - np.min(cam)
    cam_img = cam / np.max(cam)
    
    # Apply thresholding: Set values below the threshold to 0
    cam_img[cam_img < threshold] = 0

    # Resize the CAM to match the input image size (use bilinear interpolation)
    cam_img = cv2.resize(cam_img, target_size, interpolation=cv2.INTER_LINEAR)

    return cam_img

# Get weights from the fully connected layer
weight_softmax_params = list(model.fc2.parameters())
weight_softmax = np.squeeze(weight_softmax_params[0].cpu().data.numpy())

# Get the top class index
class_idx = top_class_idx.item()

# Resize CAM to match input image size (e.g., 75x100 from your input shape)
input_image_size = (100, 75)  # (width, height)

# Generate the CAM overlay with a contrast threshold (keep deeply colored regions)
cam_overlay = getCAM(activated_features.features, weight_softmax, class_idx, input_image_size, threshold=0.6)

# Load the original image (revert to HWC format for display)
original_image = np.transpose(x_validate[0].cpu().numpy(), (1, 2, 0))  # Convert back to HWC format
original_image = original_image - np.min(original_image)
original_image = original_image / np.max(original_image)  # Normalize to [0, 1]

# Plot the normal image and the heatmap overlay side by side
fig, ax = plt.subplots(1, 2, figsize=(10, 5))

# Show the original image
ax[0].imshow(original_image)
ax[0].axis('off')
ax[0].set_title("Original Image")

# Adjust transparency and colormap for better visualization
alpha_value = 0.4
  # Lower transparency for a better overlay
ax[1].imshow(original_image)
ax[1].imshow(cam_overlay, alpha=alpha_value, cmap='viridis')  # Try different colormaps like 'plasma' or 'viridis'
ax[1].axis('off')
ax[1].set_title("Image with Heatmap (Thresholded)")

plt.show()