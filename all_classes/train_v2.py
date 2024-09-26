import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from tqdm import tqdm  # For progress bars
import copy

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
IMG_HEIGHT = 224  # Updated to 224 for ResNet
IMG_WIDTH = 224   # Updated to 224 for ResNet
BATCH_SIZE = 16   # Adjusted batch size for better GPU utilization
NUM_CLASSES = 7   # Updated to 7 for all disease classes
EPOCHS = 150

'''
Training dataset size: 8011
Validation dataset size: 1002
Test dataset size: 1002
Train class counts: Counter({5: 5363, 4: 891, 2: 879, 1: 411, 0: 261, 6: 114, 3: 92})
Class weights: [4.384783798576902, 2.78449774070212, 1.3019665203965545, 12.43944099378882, 1.284431617764951, 0.21339335659678751, 10.038847117794486]
'''

class ResNet18WithDropout(nn.Module):
    def __init__(self, num_classes=7, dropout_prob=0.5):
        super(ResNet18WithDropout, self).__init__()
        self.resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        num_ftrs = self.resnet.fc.in_features
        self.resnet.fc = nn.Sequential(
            nn.Dropout(p=dropout_prob),
            nn.Linear(num_ftrs, num_classes)
        )
    
    def forward(self, x):
        return self.resnet(x)

def main():
    # Set paths to training, validation, and test directories
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
    
    print(f"Class weights(normal): {class_weights}")
    # Convert class_weights to a tensor
    class_weights = torch.tensor(class_weights, dtype=torch.float).to(device)
    print(f"Class weights(torch): {class_weights}")

    # Data loaders with num_workers set to 0
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    # Initialize the model with dropout
    model = ResNet18WithDropout(num_classes=NUM_CLASSES, dropout_prob=0.5).to(device)

    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-4)
    scheduler = StepLR(optimizer, step_size=30, gamma=0.1)

    # Training and validation loop
    train_losses, val_losses = [], []
    train_accuracies, val_accuracies = [], []
    learning_rates = []  # Initialize list to store learning rates

    best_val_acc = 0.0
    best_model_wts = copy.deepcopy(model.state_dict())
    patience = 15
    counter = 0

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

        # Update the scheduler
        scheduler.step()

        # Get current learning rate
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Current Learning Rate: {current_lr}")
        learning_rates.append(current_lr)

        # Early Stopping
        if val_epoch_acc > best_val_acc:
            best_val_acc = val_epoch_acc
            best_model_wts = copy.deepcopy(model.state_dict())
            counter = 0
            print("New best model found!")
        else:
            counter += 1
            print(f"No improvement for {counter} epochs.")
            if counter >= patience:
                print("Early stopping triggered.")
                break

    print("\nTraining complete.")

    # Load best model weights
    model.load_state_dict(best_model_wts)

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
    #plt.show()

    # Plot training & validation accuracy values
    plt.figure(figsize=(8, 4))
    plt.plot(range(1, len(train_accuracies)+1), train_accuracies, label='Train')
    plt.plot(range(1, len(val_accuracies)+1), val_accuracies, label='Validation')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig('model_accuracy.png')
    #plt.show()

    # Plot training & validation loss values
    plt.figure(figsize=(8, 4))
    plt.plot(range(1, len(train_losses)+1), train_losses, label='Train')
    plt.plot(range(1, len(val_losses)+1), val_losses, label='Validation')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('model_loss.png')
    #plt.show()

    # Plot learning rate over epochs
    plt.figure(figsize=(8, 4))
    plt.plot(range(1, len(learning_rates)+1), learning_rates, marker='o', label='Learning Rate')
    plt.title('Learning Rate Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Learning Rate')
    plt.xticks(range(1, len(learning_rates)+1))
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig('learning_rate.png')
    #plt.show()

    # Save the trained model
    os.makedirs('../weights', exist_ok=True)
    torch.save(model.state_dict(), '../weights/resnet18_ham10000_7classes_best.pth')
    print("Best model saved to '../weights/resnet18_ham10000_7classes_best.pth'")

if __name__ == '__main__':
    main()