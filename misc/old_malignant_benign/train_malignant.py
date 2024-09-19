import os
import numpy as np
import pandas as pd
from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import StratifiedGroupKFold
from PIL import Image
from matplotlib import pyplot as plt
import seaborn as sns  # For visualization

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
import torchvision.transforms.functional as F

# Set device to MPS if available (for Mac M1)
device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
print(f"Using device: {device}")

# Hyperparameters
FRACTION = 0.99
batch_size = 256
num_epochs = 2
learning_rate = 1e-4
desired_ratio = 3  # Number of negatives per positive

# Directories
DIR = '../datasets/isic_2024'
DIR_IMG = os.path.join(DIR, 'train-image', 'image')

# Load and preprocess metadata
print("Loading metadata...")
df = pd.read_csv(os.path.join(DIR, 'train-metadata.csv'), low_memory=False)
df = df.ffill()  # Fill NaN values with the previous value
print("Metadata loaded.")

# Subset for fast testing
df = df.sample(frac=FRACTION, random_state=42).reset_index(drop=True)
print(f"Using {len(df)} samples from the dataset for training.")

# Specify the test set fraction
test_fraction = 0.1  # 10% of the dataset for testing
train_val_df, test_df = np.split(
    df.sample(frac=1, random_state=42),
    [int((1 - test_fraction) * len(df))]
)
train_val_df = train_val_df.reset_index(drop=True)  # Resetting index
test_df = test_df.reset_index(drop=True)
print(f"Training on {len(train_val_df)} samples, testing on {len(test_df)} samples.")

# Class distribution
classes = np.unique(train_val_df['target'])
num_classes = len(classes)
print(f"Number of classes: {num_classes}")

# Stratified Group K-Fold split
train_val_df["fold"] = -1
sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=12)
for fold, (_, val_idx) in enumerate(
    sgkf.split(train_val_df, train_val_df.target, groups=train_val_df.patient_id)
):
    train_val_df.loc[val_idx, "fold"] = fold

# Use the first fold for validation
train_df = train_val_df[train_val_df.fold != 0].reset_index(drop=True)
val_df = train_val_df[train_val_df.fold == 0].reset_index(drop=True)
print(f"Training on {len(train_df)} samples, validating on {len(val_df)} samples.")

# Check for NaN values in the training DataFrame
nan_count = train_df.isna().sum().sum()
print(f"Total NaN values in training set: {nan_count}")

# Drop rows with NaN values
train_df = train_df.dropna().reset_index(drop=True)
print(f"Training set after dropping NaNs: {len(train_df)} samples")

# Handling Class Imbalance by Undersampling the Majority Class
print("Handling class imbalance by undersampling the majority class...")
# Separate majority and minority classes
positive_df = train_df[train_df['target'] == 1]
negative_df = train_df[train_df['target'] == 0]

num_positive = len(positive_df)
desired_negatives = num_positive * desired_ratio  # e.g., 3 negatives per positive

# Ensure we don't sample more negatives than available
available_negatives = len(negative_df)
desired_negatives = min(desired_negatives, available_negatives)

# Undersample the majority class
negative_df_sampled = negative_df.sample(n=desired_negatives, random_state=42)

# Combine minority class with sampled majority class
train_df_balanced = pd.concat([positive_df, negative_df_sampled], axis=0).sample(frac=1.0, random_state=42).reset_index(drop=True)

print(f"Balanced training data shape: {train_df_balanced.shape}")
print(f"Number of positive cases in training set: {len(positive_df)}")
print(f"Number of negative cases in training set after undersampling: {len(negative_df_sampled)}")

# Recompute class weights after balancing
print("Recomputing class weights after balancing...")
classes_resampled = np.unique(train_df_balanced['target'])
class_weights = compute_class_weight(
    class_weight='balanced',
    classes=classes_resampled,
    y=train_df_balanced['target']
)
class_weights = torch.tensor(class_weights, dtype=torch.float).to(device)
print("Class weights computed:", class_weights)

# Custom Dataset
class ISICDataset(Dataset):
    def __init__(self, df, img_dir, transform=None):
        self.df = df.reset_index(drop=True)
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_name = self.df.loc[idx, 'isic_id']
        img_path = os.path.join(self.img_dir, f"{img_name}.jpg")
        image = Image.open(img_path).convert('RGB')
        label = self.df.loc[idx, 'target']

        if self.transform:
            image = self.transform(image)

        return image, label

# Data transformations with augmentation
print("Defining data transformations...")
transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(20),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],  # ImageNet means
                         std=[0.229, 0.224, 0.225])    # ImageNet stds
])
print("Data transformations defined.")

def main():
    print("Creating datasets and dataloaders...")
    train_dataset = ISICDataset(train_df_balanced, DIR_IMG, transform=transform)
    val_dataset = ISICDataset(val_df, DIR_IMG, transform=transform)
    test_dataset = ISICDataset(test_df, DIR_IMG, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    print("Datasets and dataloaders created.")

    # Model setup
    print("Setting up the model...")
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model = model.to(device)
    print("Model setup complete.")

    # Loss and optimizer
    print("Setting up loss function and optimizer...")
    # Use class weights in the loss function
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    print("Loss function and optimizer set up.")

    # Initialize lists to store epoch metrics
    train_losses = []
    train_accuracies = []
    val_losses = []
    val_accuracies = []

    print(f"Starting training for {num_epochs} epochs...")

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        correct = 0

        print(f"\nEpoch {epoch+1}/{num_epochs} - Training...")
        for batch_idx, (images, labels) in enumerate(train_loader):
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            correct += torch.sum(preds == labels)

            # Calculate batch accuracy
            batch_acc = (preds == labels).sum().item() / labels.size(0) * 100  # Batch accuracy in percentage
            if (batch_idx + 1) % 10 == 0:  # Print every 10 batches
                print(f"Batch {batch_idx + 1}/{len(train_loader)}, Loss: {loss.item():.4f}, Batch Accuracy: {batch_acc:.2f}%")

        epoch_loss = total_loss / len(train_dataset)
        epoch_acc = correct.float() / len(train_dataset)
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.4f}")

        # Store epoch metrics
        train_losses.append(epoch_loss)
        train_accuracies.append(epoch_acc.item())  # Convert to Python float

        # Validation loop
        model.eval()
        val_loss = 0.0
        val_correct = 0

        print(f"Epoch {epoch + 1}/{num_epochs} - Validating...")
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                labels = labels.to(device)

                outputs = model(images)
                loss = criterion(outputs, labels)

                val_loss += loss.item() * images.size(0)
                _, preds = torch.max(outputs, 1)
                val_correct += torch.sum(preds == labels)

        val_epoch_loss = val_loss / len(val_dataset)
        val_epoch_acc = val_correct.float() / len(val_dataset)
        print(f"Validation Loss: {val_epoch_loss:.4f}, Validation Accuracy: {val_epoch_acc:.4f}")

        # Store validation metrics
        val_losses.append(val_epoch_loss)
        val_accuracies.append(val_epoch_acc.item())  # Convert to Python float

    print("Training complete.")

    # Evaluate on test set
    model.eval()
    test_loss = 0.0
    test_correct = 0

    print("Evaluating on test set...")
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            test_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            test_correct += torch.sum(preds == labels)

    test_epoch_loss = test_loss / len(test_dataset)
    test_epoch_acc = test_correct.float() / len(test_dataset)
    print(f"Test Loss: {test_epoch_loss:.4f}, Test Accuracy: {test_epoch_acc:.4f}")

    # Plot training & validation accuracy values
    plt.figure(figsize=(8, 4))
    plt.plot(range(1, num_epochs + 1), train_accuracies, label='Train')
    plt.plot(range(1, num_epochs + 1), val_accuracies, label='Validation')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig('malignant_benign_accuracy.png')
    plt.show()

    # Plot training & validation loss values
    plt.figure(figsize=(8, 4))
    plt.plot(range(1, num_epochs + 1), train_losses, label='Train')
    plt.plot(range(1, num_epochs + 1), val_losses, label='Validation')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('malignant_benign_loss.png')
    plt.show()

    # Save model weights
    torch.save(model.state_dict(), '../weights/malignant_benign.pth')
    print("Model weights saved.")

if __name__ == '__main__':
    main()