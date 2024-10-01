import argparse
import os
import copy
import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms

def parse_args():
    parser = argparse.ArgumentParser(description='Train EfficientNet-B5 on a specified dataset.')

    parser.add_argument('--data_dir', type=str, required=True,
                        help='Path to the dataset directory containing train, validation, and test folders.')
    parser.add_argument('--model_name', type=str, default='efficientnet_b5',
                        help='Name of the model to use. Default is efficientnet_b5.')
    parser.add_argument('--num_classes', type=int, required=True,
                        help='Number of classes in the dataset.')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Batch size for training. Default is 16.')
    parser.add_argument('--num_epochs', type=int, default=25,
                        help='Number of training epochs. Default is 25.')
    parser.add_argument('--learning_rate', type=float, default=1e-4,
                        help='Learning rate. Default is 1e-4.')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='Weight decay (L2 regularization). Default is 1e-4.')
    parser.add_argument('--use_gpu', action='store_true',
                        help='Flag to use GPU for training.')
    parser.add_argument('--save_dir', type=str, default='models',
                        help='Directory to save the trained models. Default is "models".')
    parser.add_argument('--save_every', type=int, default=5,
                        help='Save the model every N epochs. Default is 5.')
    parser.add_argument('--pretrained', action='store_true',
                        help='Use pre-trained weights.')
    parser.add_argument('--scheduler_step_size', type=int, default=7,
                        help='Step size for learning rate scheduler. Default is 7.')
    parser.add_argument('--scheduler_gamma', type=float, default=0.1,
                        help='Gamma for learning rate scheduler. Default is 0.1.')

    args = parser.parse_args()
    return args

def get_data_loaders(data_dir, batch_size):
    # Define transformations for training, validation, and testing
    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize((456, 456)),  # EfficientNet-B5 expects larger images
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(20),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],   # ImageNet means
                                 [0.229, 0.224, 0.225])   # ImageNet stds
        ]),
        'validation': transforms.Compose([
            transforms.Resize((456, 456)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
        ]),
        'test': transforms.Compose([
            transforms.Resize((456, 456)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
        ]),
    }

    image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                              data_transforms[x])
                      for x in ['train', 'validation', 'test']}

    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size,
                                                 shuffle=True if x == 'train' else False,
                                                 num_workers=4)
                  for x in ['train', 'validation', 'test']}

    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'validation', 'test']}
    class_names = image_datasets['train'].classes

    return dataloaders, dataset_sizes, class_names

def initialize_model(model_name, num_classes, pretrained=True, use_gpu=False):
    model = None
    input_size = 0

    if model_name == "efficientnet_b5":
        model = models.efficientnet_b5(pretrained=pretrained)
        num_ftrs = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(num_ftrs, num_classes)
        input_size = 456  # EfficientNet-B5 default input size
    else:
        raise ValueError("Unsupported model_name. Choose from 'efficientnet_b5'.")

    if use_gpu:
        model = model.cuda()

    return model, input_size

def train_model(model, dataloaders, dataset_sizes, criterion, optimizer, scheduler, device, num_epochs=25, save_dir='models', save_every=5):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    os.makedirs(save_dir, exist_ok=True)

    for epoch in range(1, num_epochs + 1):
        print(f'Epoch {epoch}/{num_epochs}')
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'validation']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data with progress bar
            progress_bar = tqdm(dataloaders[phase], desc=f'{phase.capitalize()} Phase', unit='batch')
            for inputs, labels in progress_bar:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # Zero the parameter gradients
                optimizer.zero_grad()

                # Forward pass
                # Track gradients only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # Backward + optimize only in train
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # Statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

                # Update progress bar
                progress_bar.set_postfix({'loss': loss.item()})

            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print(f'{phase.capitalize()} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            # Deep copy the model if it's the best so far
            if phase == 'validation' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        # Save the model at specified intervals
        if epoch % save_every == 0:
            save_path = os.path.join(save_dir, f'{model_name}_epoch_{epoch}.pth')
            torch.save(model.state_dict(), save_path)
            print(f'Model saved to {save_path}')

        print()

    time_elapsed = time.time() - since
    print(f'Training complete in {int(time_elapsed // 60)}m {int(time_elapsed % 60)}s')
    print(f'Best Validation Acc: {best_acc:.4f}')

    # Load best model weights
    model.load_state_dict(best_model_wts)
    return model

def evaluate_model(model, dataloader, dataset_size, device, class_names):
    model.eval()
    running_corrects = 0
    all_preds = []
    all_labels = []

    progress_bar = tqdm(dataloader, desc='Testing Phase', unit='batch')
    with torch.no_grad():
        for inputs, labels in progress_bar:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            running_corrects += torch.sum(preds == labels.data)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    acc = running_corrects.double() / dataset_size
    print(f'Test Acc: {acc:.4f}')

    # Optional: Add more evaluation metrics (e.g., confusion matrix, classification report)
    from sklearn.metrics import classification_report, confusion_matrix

    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds, target_names=class_names))

    print("Confusion Matrix:")
    cm = confusion_matrix(all_labels, all_preds)
    print(cm)

    # Plot Confusion Matrix
    plt.figure(figsize=(8,6))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)

    thresh = cm.max() / 2.
    for i, j in np.ndindex(cm.shape):
        plt.text(j, i, format(cm[i, j], 'd'),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig('confusion_matrix.png')
    plt.close()
    print("Confusion matrix saved as 'confusion_matrix.png'.")

def main():
    args = parse_args()

    # Determine device
    device = torch.device("cuda:0" if (args.use_gpu and torch.cuda.is_available()) else "cpu")
    print(f'Using device: {device}')

    # Get data loaders
    dataloaders, dataset_sizes, class_names = get_data_loaders(args.data_dir, args.batch_size)
    print(f'Classes: {class_names}')
    print(f'Dataset sizes: {dataset_sizes}')

    # Initialize the model
    model, input_size = initialize_model(args.model_name, args.num_classes, pretrained=args.pretrained, use_gpu=(device.type == 'cuda'))
    model = model.to(device)

    # Define loss function, optimizer, and scheduler
    criterion = nn.CrossEntropyLoss()

    # Observe that all parameters are being optimized
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

    # Decay LR by a factor of scheduler_gamma every scheduler_step_size epochs
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.scheduler_step_size, gamma=args.scheduler_gamma)

    # Train and validate
    trained_model = train_model(model, dataloaders, dataset_sizes, criterion, optimizer, scheduler, device,
                                num_epochs=args.num_epochs, save_dir=args.save_dir, save_every=args.save_every)

    # Save the best model
    best_model_path = os.path.join(args.save_dir, f'best_{args.model_name}.pth')
    torch.save(trained_model.state_dict(), best_model_path)
    print(f'Best model saved to {best_model_path}')

    # Evaluate on test set
    evaluate_model(trained_model, dataloaders['test'], dataset_sizes['test'], device, class_names)

if __name__ == '__main__':
    main()