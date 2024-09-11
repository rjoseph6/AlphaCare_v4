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

# Initialize the model architecture
model = SimpleCNN(num_classes=num_classes).to(device)
# Load the state_dict into the model
model.load_state_dict(torch.load('model_v3.pth', weights_only=True))
# Set the model to evaluation mode
#model.eval()
print("Model loaded successfully!")

# Function to perform Monte Carlo Dropout inference
def mc_dropout_inference(model, x, num_samples=50):
    model.train()  # Keep dropout layers active (necessary for Monte Carlo Dropout)
    outputs = []

    for _ in range(num_samples):
        with torch.no_grad():
            outputs.append(F.softmax(model(x), dim=1).unsqueeze(0))  # Collect predictions with softmax

    outputs = torch.cat(outputs, dim=0)  # Stack outputs for all samples
    mean_output = torch.mean(outputs, dim=0)  # Mean prediction (average over multiple runs)
    variance_output = torch.var(outputs, dim=0)  # Variance (uncertainty)

    return mean_output, variance_output

# Convert lesion_type_dict to an index-based list
class_names = list(lesion_type_dict.values())

# Perform Monte Carlo Dropout Inference on one test sample (e.g., the first sample)
test_sample = x_validate[0].unsqueeze(0)  # Select the first test sample and add a batch dimension

with torch.no_grad():
    
    # Perform Monte Carlo Dropout Inference on the selected test sample
    mean_output, uncertainty = mc_dropout_inference(model, test_sample, num_samples=50)

    # Get the predicted class index for the sample
    predicted_class_idx = torch.argmax(mean_output, dim=1).item()
    predicted_class_name = class_names[predicted_class_idx]

    # Print all class probabilities for this sample
    print("All Class Probabilities:")
    for class_idx, class_name in enumerate(class_names):
        probability = mean_output[0][class_idx].item()
        print(f"{class_name}, : {probability:.4f}")
    
    # Print predicted class and its corresponding probability
    predicted_class_probability = mean_output[0][predicted_class_idx].item()
    print("----------------------------------------------------------")
    print(f"Predicted Class: {predicted_class_name} (Class Index: {predicted_class_idx})")
    print(f"Predicted Class Probability: {predicted_class_probability:.4f}")
    # Print the uncertainty (variance) for the predicted class
    predicted_class_uncertainty = uncertainty[0][predicted_class_idx].item()
    print(f"Uncertainty (variance) for Predicted Class: {predicted_class_uncertainty:.4f}")
    print("----------------------------------------------------------")
