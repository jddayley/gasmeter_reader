# Here's a brief overview of the key components:

# Device Configuration: Automatically selects CUDA GPU, Mac GPU (MPS), or CPU based on availability, ensuring optimal use of available hardware.
# EarlyStopping Class: Monitors validation loss and stops training if there's no improvement, preventing overfitting. It also includes a placeholder for model checkpoint saving.
# Data Transforms and Augmentation: Initial data loading without augmentation is used to calculate dataset mean and standard deviation. Augmentation techniques such as random horizontal flips, grayscaling, and random rotations are then applied to the training dataset to improve model generalization.
# Model Setup: Utilizes DenseNet121 with a replaced classifier to fit the number of classes in your dataset. The model is moved to the selected device (GPU or CPU).
# Training Function: Includes detailed progress logging for training and validation phases, early stopping checks, and learning rate adjustments based on the performance on the validation set.
# Checkpoint Loading and Saving: Provides functionality to resume training from checkpoints, maintaining the state of the model, optimizer, and the epoch count.

import torchvision
import torch
import torchvision.transforms as transforms
import torchvision.models as models
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
from torch.utils.data import random_split
from torchvision.models import DenseNet121_Weights
# Assuming threshtransform.py and globals.py are in the same directory and properly set up
from threshtransform import ThresholdTransform
from globals import train_dataset_path, get_mean_and_std
from torch.optim.lr_scheduler import ReduceLROnPlateau

# Parameters
lr = 0.001
weight_decay = .003
batchsize = 32
epochs = 1000
num_classes = 10


# Device Configuration
def set_device():
    if torch.cuda.is_available():
        print("Using CUDA GPU")
        return torch.device("cuda:0")
    elif torch.backends.mps.is_available():
        print("Using Mac GPU")
        return torch.device("mps")
    else:
        print("Using CPU")
        return torch.device("cpu")

device = set_device()

class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0):
        """
        Initializes the EarlyStopping object.
        
        Args:
        - patience (int): How long to wait after last time validation loss improved.
        - verbose (bool): If True, prints a message for each validation loss improvement.
        - delta (float): Minimum change in the monitored quantity to qualify as an improvement.
        """
        self.patience = patience
        self.verbose = verbose
        self.delta = delta
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.counter = 0

    def __call__(self, val_loss, model):
        """
        Call method to update early stopping logic based on validation loss.
        
        Args:
        - val_loss (float): The current validation loss.
        - model (torch.nn.Module): The model being trained.
        """
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0
            self.save_checkpoint(val_loss, model)

    def save_checkpoint(self, val_loss, model):
        """
        Saves model when validation loss decrease.
        
        Args:
        - val_loss (float): The current validation loss.
        - model (torch.nn.Module): The model being trained.
        """
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} to {val_loss:.6f}). Saving model...')
        self.val_loss_min = val_loss
        # Save the checkpoint
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'best_loss': val_loss,
            'optimizer': optimizer.state_dict(),
        }, filename='best_checkpoint.pth.tar')

    def __call__(self, val_loss, model):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

# Data Transforms
# First, load the dataset without any augmentation to calculate mean and std
initial_transforms = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])
initial_dataset = torchvision.datasets.ImageFolder(root=train_dataset_path, transform=initial_transforms)
initial_loader = torch.utils.data.DataLoader(dataset=initial_dataset, batch_size=batchsize, shuffle=False)
mean, std = get_mean_and_std(initial_loader)  # Implement this function as needed

# Now, define transforms with data augmentation for training
train_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    #transforms.RandomHorizontalFlip(),
    transforms.Grayscale(num_output_channels=3),
    #transforms.RandomRotation(10),
    transforms.ToTensor(),
    ThresholdTransform(thr_255=75),  # Assuming custom implementation
    transforms.Normalize(torch.Tensor(mean), torch.Tensor(std))
])

# Dataset Loading and Splitting
full_dataset = torchvision.datasets.ImageFolder(root=train_dataset_path, transform=train_transforms)
train_size = int(0.8 * len(full_dataset))
valid_size = len(full_dataset) - train_size
train_dataset, valid_dataset = random_split(full_dataset, [train_size, valid_size], generator=torch.Generator().manual_seed(42))

# DataLoaders
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batchsize, shuffle=True)
valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=batchsize, shuffle=False)

# Model Setup - Using DenseNet
densenet_model = models.densenet121(weights=DenseNet121_Weights.DEFAULT)
densenet_model.classifier = nn.Linear(densenet_model.classifier.in_features, num_classes)
densenet_model.to(device)

# Loss Function and Optimizer
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(densenet_model.parameters(), lr=lr, weight_decay=weight_decay)
scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.1, patience=10, verbose=True)

# Checkpoint Functions
def save_checkpoint(state, filename="checkpoint.pth.tar"):
    torch.save(state, filename)

def load_checkpoint(filename="checkpoint.pth.tar"):
    if os.path.isfile(filename):
        print("=> Loading checkpoint")
        checkpoint = torch.load(filename)
        densenet_model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        return checkpoint['epoch'], checkpoint['best_loss']
    else:
        print("=> No checkpoint found at '{}'".format(filename))
        return 0, float('inf')

# Initialize EarlyStopping
early_stopping = EarlyStopping(patience=10, verbose=True)

def train_model(model, train_loader, valid_loader, criterion, optimizer, scheduler, num_epochs, early_stopping):
    global epoch  # If you're using epoch within EarlyStopping, ensure it's accessible
    start_epoch, best_loss = load_checkpoint()  # Load checkpoint if exists
    early_stopping.val_loss_min = best_loss  # Ensure early_stopping is aware of the best loss from a potentially loaded checkpoint
    print(f"Starting training from epoch {start_epoch+1} with best loss {best_loss:.4f}")

    for epoch in range(start_epoch, num_epochs):
        model.train()
        total_train_loss = 0
        total_train_correct = 0

        for batch_idx, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total_train_correct += (predicted == labels).sum().item()

            if batch_idx % 10 == 0:
                print(f'Epoch {epoch+1}, Batch {batch_idx}, Loss: {loss.item():.6f}')

        train_loss = total_train_loss / len(train_loader.dataset)
        train_acc = total_train_correct / len(train_loader.dataset)

        model.eval()
        total_valid_loss = 0
        total_valid_correct = 0
        with torch.no_grad():
            for inputs, labels in valid_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                total_valid_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs.data, 1)
                total_valid_correct += (predicted == labels).sum().item()

        valid_loss = total_valid_loss / len(valid_loader.dataset)
        valid_acc = total_valid_correct / len(valid_loader.dataset)

        print(f'Epoch {epoch+1}: Training Loss: {train_loss:.4f}, Validation Loss: {valid_loss:.4f}, Training Acc: {train_acc:.2f}, Validation Acc: {valid_acc:.2f}')

        scheduler.step(valid_loss)

        # Check early stopping after each epoch
        early_stopping(valid_loss, model)
        if early_stopping.early_stop:
            print("Early stopping triggered")
            break
        
        # Save checkpoint if validation loss decreased
        if valid_loss <= early_stopping.val_loss_min:
            print("Validation loss decreased, saving checkpoint...")
            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'best_loss': valid_loss,
                'optimizer': optimizer.state_dict(),
            }, filename='best_checkpoint.pth.tar')
            early_stopping.save_checkpoint(valid_loss, model)  # Update best loss within early stopping

 

        # Optional: Save model every epoch or based on condition
        # save_checkpoint({'epoch': epoch + 1, 'state_dict': model.state_dict(), 'best_loss': best_loss, 'optimizer' : optimizer.state_dict()}, filename=f"checkpoint_epoch_{epoch+1}.pth.tar")


# Start Training
train_model(densenet_model, train_loader, valid_loader, loss_fn, optimizer, scheduler, epochs, early_stopping)
