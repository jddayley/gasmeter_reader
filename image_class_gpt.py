# Here's a brief overview of the key components:

# Device Configuration: Automatically selects CUDA GPU, Mac GPU (MPS), or CPU based on availability, ensuring optimal use of available hardware.
# EarlyStopping Class: Monitors validation loss and stops training if there's no improvement, preventing overfitting. It also includes a placeholder for model checkpoint saving.
# Data Transforms and Augmentation: Initial data loading without augmentation is used to calculate dataset mean and standard deviation. Augmentation techniques such as random horizontal flips, grayscaling, and random rotations are then applied to the training dataset to improve model generalization.
# Model Setup: Utilizes DenseNet121 with a replaced classifier to fit the number of classes in your dataset. The model is moved to the selected device (GPU or CPU).
# Training Function: Includes detailed progress logging for training and validation phases, early stopping checks, and learning rate adjustments based on the performance on the validation set.
# Checkpoint Loading and Saving: Provides functionality to resume training from checkpoints, maintaining the state of the model, optimizer, and the epoch count.

import torch
import torchvision
from torchvision import transforms, datasets
from torch.utils.data import DataLoader, random_split
import torchvision.models as models
import torch.nn as nn
import torch.optim as optim
import os
import numpy as np
from threshtransform import ThresholdTransform

# Configuration
config = {
    "lr": 0.001,
    "weight_decay": 0.003,
    "batch_size": 32,
    "epochs": 100,
    "num_classes": 10,
    "patience": 10,
    "device_preference": "auto",
    "train_dataset_path": "data/images_copy",
    "checkpoint_path": "checkpoints/checkpoint.pth.tar"
}

def set_device(device_preference="auto"):
    if device_preference == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    elif device_preference == "cpu":
        return torch.device("cpu")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")

class EarlyStopping:
    def __init__(self, patience=10, verbose=True, delta=0, path='checkpoints/checkpoint.pth.tar'):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        self.delta = delta
        self.path = path

    def __call__(self, val_loss, model, optimizer, epoch):
        if self.best_loss is None:
            self.best_loss = val_loss
            self.save_checkpoint(val_loss, model, optimizer, epoch)
        elif val_loss < self.best_loss - self.delta:
            self.best_loss = val_loss
            self.save_checkpoint(val_loss, model, optimizer, epoch)
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

    def save_checkpoint(self, val_loss, model, optimizer, epoch):
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        if self.verbose:
            print(f'Validation loss decreased to {val_loss:.6f}. Saving model...')
        torch.save({
            'epoch': epoch,
            'model_state': model.state_dict(),
            'optimizer_state': optimizer.state_dict(),
            'loss': val_loss,
        }, self.path)

def calculate_mean_std(loader):
    mean = 0.
    std = 0.
    total_images_count = 0
    for images, _ in loader:
        images = images.view(images.size(0), images.size(1), -1)
        mean += images.mean(2).sum(0)
        std += images.std(2).sum(0)
        total_images_count += 1
    mean /= total_images_count
    std /= total_images_count
    return mean, std

def initialize_model(num_classes, device):
    model = models.densenet121(pretrained=False)
    model.classifier = nn.Linear(model.classifier.in_features, num_classes)
    model.to(device)
    return model

def get_data_transforms(mean, std):
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        ThresholdTransform(thr_255=75),
        transforms.Normalize(mean=mean, std=std)
    ])

def prepare_datasets(batch_size, train_transforms):
    full_dataset = datasets.ImageFolder(root=config["train_dataset_path"], transform=train_transforms)
    train_size = int(0.8 * len(full_dataset))
    valid_size = len(full_dataset) - train_size
    train_dataset, valid_dataset = random_split(full_dataset, [train_size, valid_size], generator=torch.Generator().manual_seed(42))
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, valid_loader


def train_one_epoch(model, device, train_loader, optimizer, criterion):
    model.train()
    total_loss, total_correct = 0, 0
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * inputs.size(0)
        _, predicted = torch.max(outputs.data, 1)
        total_correct += (predicted == labels).sum().item()
    avg_loss = total_loss / len(train_loader.dataset)
    accuracy = total_correct / len(train_loader.dataset)
    print(f'Train Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}')
    return avg_loss, accuracy

def validate_model(model, device, valid_loader, criterion):
    model.eval()
    total_loss, total_correct = 0, 0
    with torch.no_grad():
        for inputs, labels in valid_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            total_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total_correct += (predicted == labels).sum().item()
    avg_loss = total_loss / len(valid_loader.dataset)
    accuracy = total_correct / len(valid_loader.dataset)
    print(f'Validation Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}')
    return avg_loss, accuracy

def main():
    device = set_device(config["device_preference"])

    initial_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    initial_dataset = datasets.ImageFolder(root=config["train_dataset_path"], transform=initial_transforms)
    initial_loader = DataLoader(initial_dataset, batch_size=config["batch_size"], shuffle=False)
    mean, std = calculate_mean_std(initial_loader)

    model = initialize_model(config["num_classes"], device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, verbose=True)

    train_transforms = get_data_transforms(mean.numpy(), std.numpy())
    train_loader, valid_loader = prepare_datasets(config["batch_size"], train_transforms)

    early_stopping = EarlyStopping(patience=config["patience"], verbose=True, path=config["checkpoint_path"])

    for epoch in range(config["epochs"]):
        train_loss, train_acc = train_one_epoch(model, device, train_loader, optimizer, criterion)
        valid_loss, valid_acc = validate_model(model, device, valid_loader, criterion)
        print(f'Epoch {epoch}, Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Valid Loss: {valid_loss:.4f}, Valid Acc: {valid_acc:.4f}')

        scheduler.step(valid_loss)

        if early_stopping.early_stop:
            print("Early stopping triggered.")
            break
        else:
            early_stopping(valid_loss, model, optimizer, epoch)

if __name__ == "__main__":
    main()
