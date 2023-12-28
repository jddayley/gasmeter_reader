
# Choose device
import torchvision
import torch
import torchvision.transforms as transforms  
import torchvision.models as models
import torch.nn as nn
import torch.optim as optim
import os
import matplotlib.pyplot as plt
import numpy as np
from threshtransform import *
from globals import *
from torch.optim.lr_scheduler import StepLR
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import random_split
from torchvision.models import DenseNet121_Weights
import sys
# Parameters
lr = 0.001
momentum = 0.9
weight_decay = .003
batchsize = 32
epochs = 400
num_classes = 10
train_transforms =  transforms.Compose([transforms.Resize((224,224)),
    transforms.ToTensor()
])
train_dataset = torchvision.datasets.ImageFolder( root= train_dataset_path, transform=train_transforms)
train_loader = torch.utils.data.DataLoader (dataset= train_dataset, batch_size=batchsize, shuffle=False)
mean, std = get_mean_and_std(train_loader)
print ("Mean: " )
print(mean)
print ("Std")
print(std)
# Data Transforms
train_transforms = transforms.Compose([
    
    transforms.Resize((224,224)),
    #transforms.RandomHorizontalFlip(),
    transforms.Grayscale(num_output_channels=3),
    #transforms.RandomRotation(10),
    transforms.ToTensor(),
    ThresholdTransform(thr_255=75),
    transforms.Normalize(torch.Tensor(mean), torch.Tensor(std))
    # transforms.Normalize((0.5), (0.5))
    ])

# Device Configuration
def set_device():
    if torch.cuda.is_available():
        print ("Using CUDA GPU")
        dev = "cuda:0"
    elif torch.backends.mps.is_available():
        print ("Using Mac GPU")
        dev = "mps"
    else:
        print ("Using CPU")
        dev = "cpu"
    return torch.device(dev)

# Set a fixed seed for reproducibility
torch.manual_seed(42)  # You can choose any number as your seed

# Dataset Loading and Splitting
dataset_path = train_dataset_path  # Replace with your dataset path
full_dataset = torchvision.datasets.ImageFolder(root=dataset_path, transform=train_transforms)

# Consistent Splitting of Dataset
train_size = int(0.8 * len(full_dataset))
valid_size = len(full_dataset) - train_size
train_dataset, valid_dataset = random_split(full_dataset, [train_size, valid_size], generator=torch.Generator().manual_seed(42))

# DataLoaders
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batchsize, shuffle=True)
valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=batchsize, shuffle=False)
# Model Setup - Using DenseNet
weights = DenseNet121_Weights.DEFAULT  # or choose another variant if needed
densenet_model = models.densenet121(weights=weights)


# Replace the classifier
num_ftrs = densenet_model.classifier.in_features
densenet_model.classifier = nn.Linear(num_ftrs, num_classes)  # Adjust 'num_classes' to your dataset

# Device Configuration and Model Transfer
device = set_device()
densenet_model.to(device)

# Loss Function and Optimizer
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(densenet_model.parameters(), lr=lr, weight_decay=weight_decay)
# device = set_device()
# resnet50_model.to(device)
# loss_fn = nn.CrossEntropyLoss()

# Using Adam Optimizer
#optimizer = optim.Adam(resnet50_model.parameters(), lr=lr, weight_decay=weight_decay)

# Model Setup
# resnet50_model = models.resnet50()
# #resnet50_model = models.resnet50(pretrained=True)
# num_ftrs = resnet50_model.fc.in_features
# resnet50_model.fc = nn.Linear(num_ftrs, 10)  # Adjust for your number of classes
# Learning Rate Scheduler (optional)
# Adjust or remove the scheduler as per your requirement
scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.1, patience=10, verbose=True)




# # Loss and Optimizer
# loss_fn = nn.CrossEntropyLoss()
# optimizer = optim.SGD(resnet50_model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
# scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.1, patience=10, verbose=True)

# Checkpoint Functions
def save_checkpoint(state, filename="checkpoint.pth.tar"):
    torch.save(state, filename)

def load_checkpoint(filename):
    if os.path.isfile(filename):
        print("=> Loading checkpoint")
        checkpoint = torch.load(filename)
        densenet_model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        return checkpoint['epoch'], checkpoint['best_loss']
    else:
        print("=> No checkpoint found at '{}'".format(filename))
        return 0, float('inf')

# Training Function
def train_model(model, train_loader, valid_loader, criterion, optimizer, scheduler, num_epochs, start_epoch, best_loss):
    for epoch in range(start_epoch, num_epochs):
        print(f'Epoch {epoch+1}/{num_epochs}')
        model.train()  # Training mode
        total_train_loss, total_valid_loss = 0, 0
        total_train_correct, total_valid_correct = 0, 0

        # Training Phase
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total_train_correct += (predicted == labels).sum().item()

        train_loss = total_train_loss / len(train_loader.dataset)
        train_acc = total_train_correct / len(train_loader.dataset) * 100
        print(f'Training Loss: {train_loss:.4f}, Accuracy: {train_acc:.2f}%')

        # Validation Phase
        model.eval()  # Evaluation mode
        with torch.no_grad():
            for inputs, labels in valid_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                total_valid_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs.data, 1)
                total_valid_correct += (predicted == labels).sum().item()

        valid_loss = total_valid_loss / len(valid_loader.dataset)
        valid_acc = total_valid_correct / len(valid_loader.dataset) * 100
        print(f'Validation Loss: {valid_loss:.4f}, Accuracy: {valid_acc:.2f}%')

        # Learning Rate Adjustment
        scheduler.step(valid_loss)

        # Save Checkpoint
        is_best = valid_loss < best_loss
        best_loss = min(valid_loss, best_loss)
        save_checkpoint({
            'epoch': epoch + 1, 
            'state_dict': model.state_dict(), 
            'best_loss': best_loss, 
            'optimizer': optimizer.state_dict()},
            filename='best_checkpoint.pth.tar' if is_best else 'checkpoint.pth.tar'
        )
    print('Training complete')

# Start Training
start_epoch, best_loss = load_checkpoint('checkpoint.pth.tar')
train_model(densenet_model, train_loader, valid_loader, loss_fn, optimizer, scheduler, epochs, start_epoch, best_loss)
