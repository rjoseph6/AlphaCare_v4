import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, WeightedRandomSampler
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from tqdm import tqdm  # For progress bars

# ------------------------------
# Configuration
# ------------------------------

# Set device to MPS (Apple Silicon GPU) if available, else CPU
if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("Using MPS device")
else:
    device = torch.device("cpu")
    print("Using CPU device")

# Image parameters
IMG_HEIGHT = 100  # Updated to 224 for ResNet
IMG_WIDTH = 75   # Updated to 224 for ResNet
BATCH_SIZE = 32   # Adjusted batch size for better GPU utilization
NUM_CLASSES = 7   # Updated to 7 for all disease classes
EPOCHS = 2

def main():
    # Set paths to training, validation, and test directories
    # Ensure these paths point to the directories with all 7 classes
    train_dir = '../datasets/ham10000_classified/train'
    validation_dir = '../datasets/ham10000_classified/val'
    test_dir = '../datasets/ham10000_classified/test'

    # Verify that all directories exist
    for directory in [train_dir, validation_dir, test_dir]:
        if not os.path.isdir(directory):
            raise FileNotFoundError(f"Directory '{directory}' does not exist. Please check the path.")

    # Data transformations (including data augmentation for training)
    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize((IMG_HEIGHT, IMG_WIDTH)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(20),
            transforms.RandomResizedCrop(IMG_HEIGHT, scale=(0.8, 1.0)),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],   # Mean for normalization
                                 [0.229, 0.224, 0.225])  # Std for normalization
        ]),
        'val': transforms.Compose([
            transforms.Resize((IMG_HEIGHT, IMG_WIDTH)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
        ]),
        'test': transforms.Compose([  # Added separate transform for test
            transforms.Resize((IMG_HEIGHT, IMG_WIDTH)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
        ]),
    }

    # Datasets
    train_dataset = datasets.ImageFolder(root=train_dir, transform=data_transforms['train'])
    val_dataset = datasets.ImageFolder(root=validation_dir, transform=data_transforms['val'])
    test_dataset = datasets.ImageFolder(root=test_dir, transform=data_transforms['test'])

    print(f"Training dataset size: {len(train_dataset)}")
    print(f"Validation dataset size: {len(val_dataset)}")
    print(f"Test dataset size: {len(test_dataset)}")

    # Compute class weights for imbalanced dataset
    train_targets = [label for _, label in train_dataset.samples]
    class_counts = Counter(train_targets)
    print(f"Train class counts: {class_counts}")
    total_samples = sum(class_counts.values())

    # Compute class weights
    class_weights = []
    for label in range(NUM_CLASSES):
        if class_counts[label] > 0:
            class_weight = total_samples / (NUM_CLASSES * class_counts[label])
        else:
            class_weight = 0  # Avoid division by zero
            print(f"Warning: Class {label} has zero samples in training data.")
        class_weights.append(class_weight)
        print(f"Class {label} count: {class_counts[label]}")
        print(f"Class {label} weight: {class_weight}")

    print(f"Class weights: {class_weights}")

    # Compute sample weights for the dataset
    sample_weights = [class_weights[label] for _, label in train_dataset.samples]

    # Create a WeightedRandomSampler
    sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)

    # Data loaders with num_workers set appropriately
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=sampler, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    # Load pre-trained ResNet-18 model and modify for multi-class classification
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

    num_ftrs = model.fc.in_features
    # Replace the last layer with a multi-class classification layer
    model.fc = nn.Linear(num_ftrs, NUM_CLASSES)
    model = model.to(device)

    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    # Training and validation loop
    train_losses, val_losses = [], []
    train_accuracies, val_accuracies = [], []

    for epoch in range(EPOCHS):
        print(f"\nEpoch {epoch+1}/{EPOCHS}")
        print('-' * 10)
        
        # Training Phase
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        # Progress bar for training
        train_bar = tqdm(train_loader, desc='Training', leave=False)
        for inputs, labels in train_bar:
            inputs = inputs.to(device)
            labels = labels.to(device)

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # Statistics
            running_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            correct += torch.sum(preds == labels.data)
            total += labels.size(0)

            # Update progress bar
            train_bar.set_postfix(loss=loss.item())

        epoch_loss = running_loss / len(train_dataset)
        epoch_acc = correct.float() / len(train_dataset)
        train_losses.append(epoch_loss)
        train_accuracies.append(epoch_acc.item())
        print(f"Training Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")

        # Validation Phase
        model.eval()
        val_running_loss = 0.0
        val_correct = 0
        val_total = 0
        all_labels = []
        all_preds = []

        # Progress bar for validation
        val_bar = tqdm(val_loader, desc='Validation', leave=False)
        with torch.no_grad():
            for inputs, labels in val_bar:
                inputs = inputs.to(device)
                labels = labels.to(device)

                outputs = model(inputs)
                loss = criterion(outputs, labels)

                val_running_loss += loss.item() * inputs.size(0)
                _, preds = torch.max(outputs, 1)
                val_correct += torch.sum(preds == labels.data)
                val_total += labels.size(0)
                all_labels.extend(labels.cpu().numpy())
                all_preds.extend(preds.cpu().numpy())

                # Update progress bar
                val_bar.set_postfix(loss=loss.item())

        val_epoch_loss = val_running_loss / len(val_dataset)
        val_epoch_acc = val_correct.float() / len(val_dataset)
        val_losses.append(val_epoch_loss)
        val_accuracies.append(val_epoch_acc.item())
        print(f"Validation Loss: {val_epoch_loss:.4f} Acc: {val_epoch_acc:.4f}")

    print("\nTraining complete.")

    # Evaluate on test set
    model.eval()
    test_loss = 0.0
    test_correct = 0
    all_test_labels = []
    all_test_preds = []

    print("Evaluating on test set...")
    with torch.no_grad():
        test_bar = tqdm(test_loader, desc='Testing', leave=False)
        for images, labels in test_bar:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            test_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            test_correct += torch.sum(preds == labels)
            all_test_labels.extend(labels.cpu().numpy())
            all_test_preds.extend(preds.cpu().numpy())

            # Update progress bar
            test_bar.set_postfix(loss=loss.item())

    test_epoch_loss = test_loss / len(test_dataset)
    test_epoch_acc = test_correct.float() / len(test_dataset)
    print(f"Test Loss: {test_epoch_loss:.4f}, Test Accuracy: {test_epoch_acc:.4f}")

    # Generate classification report and confusion matrix for test set
    print("\nClassification Report on Test Set:")
    print(classification_report(all_test_labels, all_test_preds, target_names=train_dataset.classes))

    print("\nConfusion Matrix on Test Set:")
    cm = confusion_matrix(all_test_labels, all_test_preds)
    print(cm)

    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    plt.colorbar()
    tick_marks = np.arange(NUM_CLASSES)
    plt.xticks(tick_marks, train_dataset.classes, rotation=45)
    plt.yticks(tick_marks, train_dataset.classes)

    fmt = 'd'
    thresh = cm.max() / 2.
    for i in range(NUM_CLASSES):
        for j in range(NUM_CLASSES):
            plt.text(j, i, format(cm[i, j], fmt),
                     ha="center", va="center",
                     color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png')
    plt.show()

    # Plot training & validation accuracy values
    plt.figure(figsize=(8, 4))
    plt.plot(range(1, EPOCHS+1), train_accuracies, label='Train')
    plt.plot(range(1, EPOCHS+1), val_accuracies, label='Validation')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig('model_accuracy.png')
    plt.show()

    # Plot training & validation loss values
    plt.figure(figsize=(8, 4))
    plt.plot(range(1, EPOCHS+1), train_losses, label='Train')
    plt.plot(range(1, EPOCHS+1), val_losses, label='Validation')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('model_loss.png')
    plt.show()

    # Save the trained model
    os.makedirs('../weights', exist_ok=True)
    torch.save(model.state_dict(), '../weights/resnet18_ham10000_7classes.pth')
    print("Model saved to '../weights/resnet18_ham10000_7classes.pth'")

if __name__ == '__main__':
    main()