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
skin_df = skin_df.sample(frac=0.1) # shuffle the dataset
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

# Reshape images for PyTorch (channels-first format)
x_train = torch.tensor(x_train).permute(0, 3, 1, 2).float().to(device)  # (batch_size, channels, height, width)
x_test = torch.tensor(x_test).permute(0, 3, 1, 2).float().to(device)    # (batch_size, channels, height, width)
x_validate = torch.tensor(x_validate).permute(0, 3, 1, 2).float().to(device)  # (batch_size, channels, height, width)

y_train = y_train.to(device)
y_test = y_test.to(device)
y_validate = y_validate.to(device)

# Define SimpleCNN model
class SimpleCNN(nn.Module):
    def __init__(self, num_classes=7):
        super(SimpleCNN, self).__init__()
        
        # First convolutional layer
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.dropout1 = nn.Dropout(0.5)
        
        # Second convolutional layer
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.dropout2 = nn.Dropout(0.5)
        
        # Third convolutional layer
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.dropout3 = nn.Dropout(0.5)
        
        self.pool = nn.MaxPool2d(2, 2)  # 2x2 Max Pooling
        
        # Fully connected layers
        self.fc1 = nn.Linear(128 * 9 * 12, 512)  # Flatten after pooling
        self.dropout_fc1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.dropout1(x)
        x = self.pool(F.relu(self.conv2(x)))
        x = self.dropout2(x)
        x = self.pool(F.relu(self.conv3(x)))
        x = self.dropout3(x)
        x = x.view(-1, 128 * 9 * 12)  # Flatten
        x = F.relu(self.fc1(x))
        x = self.dropout_fc1(x)
        x = self.fc2(x)
        return x

model = SimpleCNN(num_classes=7).to(device)
model.load_state_dict(torch.load('model_v3.pth', weights_only=True))
model.eval()

# Grad-CAM Function
def generate_heatmap(model, input_image, target_class):
    grad_cam = {}
    
    def save_grad(name):
        def hook(grad):
            grad_cam[name] = grad
        return hook
    
    # Forward pass to get output
    output = model(input_image)
    
    # Backward pass to get gradients
    model.conv3.register_backward_hook(save_grad('conv3'))
    one_hot_output = torch.FloatTensor(1, output.size()[-1]).zero_().to(device)
    one_hot_output[0][target_class] = 1
    output.backward(gradient=one_hot_output)
    
    # Get gradients and activations from the third conv layer
    gradients = grad_cam['conv3'].cpu().data.numpy()[0]
    activations = model.conv3(input_image).detach().cpu().data.numpy()[0]
    
    # Compute the weighted average of gradients
    weights = np.mean(gradients, axis=(1, 2))
    heatmap = np.sum(weights[:, np.newaxis, np.newaxis] * activations, axis=0)
    heatmap = np.maximum(heatmap, 0)
    heatmap /= np.max(heatmap)
    
    return heatmap

# Generate heatmap for one sample
test_sample = x_validate[0].unsqueeze(0)
target_class = torch.argmax(model(test_sample), dim=1).item()

# Generate heatmap
heatmap = generate_heatmap(model, test_sample, target_class)

# Plot original image and heatmap
plt.imshow(test_sample[0].permute(1, 2, 0).cpu())
plt.imshow(heatmap, cmap='jet', alpha=0.5)  # Overlay heatmap on the image
plt.colorbar()
plt.title(f'Heatmap for Predicted Class {target_class}')
plt.show()