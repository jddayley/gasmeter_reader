import torchvision
import torch
import torchvision.transforms as transforms  
import torchvision.models as models
import torch.nn as nn 
import torch.optim as optim
import os
import matplotlib.pyplot as plt
import numpy as np

train_dataset_path = "/Users/ddayley/Desktop/gas/data/images"
test_dataset_path = "/Users/ddayley/Desktop/gas/data/images_test"


def get_mean_and_std(loader):
    mean = 0.
    std = 0.
    total_images_count = 0
    for images, _ in loader:
        image_count_in_a_batch = images.size(0)
        images = images.view(image_count_in_a_batch, images.size(1), -1)
        mean += images.mean(2).sum(0)
        std += images.std(2).sum(0)
        total_images_count += image_count_in_a_batch
   #     print(images.shape)
    mean /= total_images_count
    std /= total_images_count
    return mean, std
train_transforms =  transforms.Compose([transforms.Resize((224,224)),
    transforms.ToTensor()
])
train_dataset = torchvision.datasets.ImageFolder( root= train_dataset_path, transform=train_transforms)
train_loader = torch.utils.data.DataLoader (dataset= train_dataset, batch_size=32, shuffle=False)
mean, std = get_mean_and_std(train_loader)
print ("Mean: " )
print(mean)
print ("Std")
print(std)
mean = [0.4363, 0.4328, 0.3291]
std = [0.2129, 0.2075, 0.2038]
train_transforms = transforms.Compose([
    
    transforms.Resize((224,224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize(torch.Tensor(mean), torch.Tensor(std))
    # transforms.Normalize((0.5), (0.5))
    ])
test_transforms = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    # transforms.Normalize((0.5), (0.5))
    transforms.Normalize(torch.Tensor(mean), torch.Tensor(std))
    ])
print("DONE")
#print(os.listdir(train_dataset_path))
train_dataset = torchvision.datasets.ImageFolder( root= train_dataset_path, transform=train_transforms)
test_dataset = torchvision.datasets.ImageFolder( root= test_dataset_path, transform=test_transforms)

def show_transformed_images(dataset):
    loader = torch.utils.data.DataLoader(dataset,batch_size = 6, shuffle=True)
    batch = next(iter(loader))
    images, labels = batch
    grid = torchvision.utils.make_grid(images, nrow=3)
    #plt.figure(figsize=(11,11))
    #plt.imshow(np.transpose(grid, (1,2,0)))
    #plt.show()
    print('labels: ', labels)
show_transformed_images(train_dataset)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=True)

def set_device():
    if torch.cuda.is_available():
        dev = "cuda:0"
    else:
        dev  = "cpu"
    return torch.device(dev)
def train_nn(model, train_loader, test_loader, criterion, optimizer, n_epochs):
    device = set_device()
    best_acc = 0
    for epoch in range(n_epochs):
        print("Epoch number %d "% (epoch +1) )
        model.train()
        running_loss = 0.0
        running_correct = 0.0
        total = 0

        for data in train_loader:
            images, labels = data
            images = images.to(device)
            labels = labels.to(device)
            total += labels.size(0)

            optimizer.zero_grad()

            outputs = model(images)

            _, predicted  = torch.max(outputs.data, 1)

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            running_correct += (labels==predicted).sum().item()
        epoch_loss = running_loss/len(train_loader)
        epoch_acc = 100.00 * running_correct /total
        print (" - Training dataset. Got %d out of %d images correctly (%.3f%%). Epch loss: %.3f" 
        % (running_correct, total, epoch_acc, epoch_loss))
        test_dataset_acc = evaluate_model_on_test_set(model, test_loader)
        if (test_dataset_acc > best_acc):
            best_acc = test_dataset_acc
            save_checkpoint(model,epoch, optimizer, best_acc)
    print("Finished")
    return model

def evaluate_model_on_test_set(model, test_loader):
    model.eval()
    predicted_correctly_on_epoch = 0
    total = 0
    device = set_device()
    with torch.no_grad():
        for data in test_loader:
            images, labels = data
            images = images.to(device)
            labels = labels.to(device)
            total += labels.size(0)
            outputs = model(images)
            _, predicted  = torch.max(outputs.data, 1)
            predicted_correctly_on_epoch += (predicted == labels).sum().item()

    epoch_acc = 100.00 * predicted_correctly_on_epoch /total
    print (" - Testing dataset. Got %d out of %d images correctly (%.3f%%)" 
        % (predicted_correctly_on_epoch,total, epoch_acc ))
    return epoch_acc

def save_checkpoint(model,epoch, optimizer, best_acc):
    state = {
        'epoch': epoch + 1,
        'model': model.state_dict(),
        'best_accuracy' : best_acc,
        'optimizer' : optimizer.state_dict()
    }
    torch.save(state, 'model_best_checkpoint.pth.tar')
    

resnet18_model = models.resnet18(pretrained=True)
#resnet18_model = models.resnet50(weights='ResNet50_Weights.DEFAULT')
num_ftrs = resnet18_model.fc.in_features
number_of_classes = 10
resnet18_model.fc = nn.Linear(num_ftrs,number_of_classes)
device = set_device()
resnet18_model = resnet18_model.to(device)
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.SGD(resnet18_model.parameters(),lr=0.01, momentum=0.9, weight_decay=.003)
train_nn(resnet18_model, train_loader, test_loader, loss_fn, optimizer, 20)
checkpoint = torch.load('model_best_checkpoint.pth.tar')
print(checkpoint['epoch'])
print(checkpoint['best_accuracy'])
resnet18_model = models.resnet18()
num_ftrs = resnet18_model.fc.in_features
number_of_classes = 10
resnet18_model.fc = nn.Linear(num_ftrs,number_of_classes)
resnet18_model.load_state_dict(checkpoint['model'])
torch.save(resnet18_model, 'best_model.pth')