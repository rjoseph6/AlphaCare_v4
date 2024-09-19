import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from tqdm import tqdm
import os

# Check for MPS availability
if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("Using MPS (Metal Performance Shaders) device")
else:
    device = torch.device("cpu")
    print("MPS device not found. Using CPU")

# Data directory
data_dir = '../datasets/skin_nonskin'  # Replace with your actual data directory path
NUM_CLASSES = 2

# Simplified data transformations
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# Create datasets
image_datasets = {
    'train': datasets.ImageFolder(os.path.join(data_dir, 'train'), train_transform),
    'val': datasets.ImageFolder(os.path.join(data_dir, 'val'), val_transform)
}

# Create dataloaders
dataloaders = {
    'train': DataLoader(image_datasets['train'], batch_size=64, shuffle=True, num_workers=4),
    'val': DataLoader(image_datasets['val'], batch_size=64, shuffle=False, num_workers=4)
}

# Dataset sizes
dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
print(f"Training set size: {dataset_sizes['train']}")
print(f"Validation set size: {dataset_sizes['val']}")

# Class names
class_names = image_datasets['train'].classes
print(f"Class names: {class_names}")

model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
num_ftrs = model.fc.in_features
# Replace the last layer with a binary classification layer
model.fc = nn.Linear(num_ftrs, NUM_CLASSES)
model = model.to(device)

# Load pre-trained MobileNetV2 model with updated weights argument
#model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
# Modify the classifier
#model.classifier[1] = nn.Linear(model.classifier[1].in_features, 2)
#print(model)

# Move model to device
#model = model.to(device)

# Loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Number of epochs
num_epochs = 5

def train_model():
    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1}/{num_epochs}')
        print('-' * 30)
        
        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluation mode
            
            running_loss = 0.0
            running_corrects = 0
            
            # Iterate over data
            dataloader = dataloaders[phase]
            dataset_size = dataset_sizes[phase]
            
            pbar = tqdm(dataloader, desc=f'{phase.capitalize()} Phase', unit='batch')
            for inputs, labels in pbar:
                # Move inputs and labels to device
                inputs = inputs.to(device)
                labels = labels.to(device)
                
                optimizer.zero_grad()
                
                # Forward pass
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)
                    
                    # Backward and optimize in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                
                # Statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
                
                # Update progress bar
                pbar.set_postfix({'loss': loss.item()})
            
            # Calculate loss and accuracy for the epoch
            epoch_loss = running_loss / dataset_size
            epoch_acc = running_corrects.float() / dataset_size
            #epoch_acc = running_corrects.double() / dataset_size
            
            print(f'{phase.capitalize()} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
        print()
    
    # Save the trained model
    torch.save(model.state_dict(), '../weights/skin_nonskin_resnet.pth')

if __name__ == '__main__':
    
    train_model()
    print("Training completed")