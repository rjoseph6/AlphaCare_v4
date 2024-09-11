import numpy as np
import pandas as pd
import os
import torch
from tqdm import tqdm
from glob import glob
from PIL import Image
from sklearn.model_selection import train_test_split
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

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
#skin_df = skin_df.sample(frac=0.05) # shuffle the dataset
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

# Define a Basic CNN Model
class SimpleCNN(nn.Module):
    def __init__(self, num_classes=7):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)  # 2x2 Max Pooling
        self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(128 * 9 * 12, 512)  # Flatten after pooling
        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))  # First conv layer
        x = self.pool(F.relu(self.conv2(x)))  # Second conv layer
        x = self.pool(F.relu(self.conv3(x)))  # Third conv layer
        x = x.view(-1, 128 * 9 * 12)  # Flatten the tensor for fully connected layers
        x = F.relu(self.fc1(x))  # First fully connected layer
        x = self.fc2(x)  # Output layer
        return x

# Initialize the model, loss function, and optimizer
model = SimpleCNN(num_classes=num_classes).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training Loop
epochs = 25
batch_size = 512
for epoch in range(epochs):
    model.train()
    permutation = torch.randperm(x_train.size()[0])
    
    running_loss = 0.0
    correct_train = 0
    total_train = 0

    for i in range(0, x_train.size()[0], batch_size):
        indices = permutation[i:i + batch_size]
        batch_x, batch_y = x_train[indices], y_train[indices]

        optimizer.zero_grad()
        outputs = model(batch_x)
        loss = criterion(outputs, torch.max(batch_y, 1)[1])
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()

        # Calculate training accuracy
        _, predicted = torch.max(outputs, 1)
        _, labels = torch.max(batch_y, 1)
        correct_train += (predicted == labels).sum().item()
        total_train += labels.size(0)
    
    train_accuracy = 100 * correct_train / total_train
    print(f"Epoch {epoch+1}/{epochs}, Loss: {running_loss / len(x_train):.4f}, Training Accuracy: {train_accuracy:.2f}%")
    
    # Validation accuracy
    model.eval()
    correct_val = 0
    total_val = 0
    with torch.no_grad():
        outputs = model(x_validate)
        _, predicted = torch.max(outputs, 1)
        _, labels = torch.max(y_validate, 1)
        correct_val += (predicted == labels).sum().item()
        total_val += labels.size(0)
    
    val_accuracy = 100 * correct_val / total_val
    print(f'Epoch {epoch+1}/{epochs}, Validation Accuracy: {val_accuracy:.2f}%')

print('Finished Training')

# Testing the model on the test dataset
print('Evaluating on test set...')
model.eval()
correct_test = 0
total_test = 0
with torch.no_grad():
    outputs = model(x_test)
    _, predicted = torch.max(outputs, 1)
    _, labels = torch.max(y_test, 1)
    correct_test += (predicted == labels).sum().item()
    total_test += labels.size(0)

test_accuracy = 100 * correct_test / total_test
print(f'Test Accuracy: {test_accuracy:.2f}%')

# Save the model weights with test accuracy in the filename
model_filename = f'../weights/model_v1_acc_{test_accuracy:.0f}.pth'
print(f'Saving model as {model_filename}...')
torch.save(model.state_dict(), model_filename)