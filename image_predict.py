import torchvision
import torch
import torchvision.transforms as transforms
import PIL.Image as Image
import cv2 
import os
import sys
from threshtransform import *
from globals import *
import torchvision.models as models
import torch.nn as nn 
import torch.optim as optim
from torchvision.models import DenseNet121_Weights
import matplotlib.pyplot as plt

lr = 0.001
momentum = 0.9
#weight_decay = 1.0e-4
weight_decay = .003
batchsize = 32
#batchsize_valid = 64
#start_epoch = 0
epochs      = 50
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]
classes = [
        "0",
        "1",
        "2",
        "3",
        "4",
        "5",
        "6",
        '7',
        "8",
        "9"
        ]
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
def load_model():
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
#mean = [0.4868, 0.4868, 0.4868]
#std = [0.1944, 0.1944, 0.1944]
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
    test_transforms = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    ThresholdTransform(thr_255=75),
    # transforms.Normalize((0.5), (0.5))
    transforms.Normalize(torch.Tensor(mean), torch.Tensor(std))
    ])
#print("Setting up model...")
#     resnet50_model = models.resnet50(pretrained=True)
# #resnet50_model = models.resnet50()
# #resnet50_model = models.resnet50(weights='ResNet50_Weights.DEFAULT')
#     num_ftrs = resnet50_model.fc.in_features
    num_classes = 10
#     resnet50_model.fc = nn.Linear(num_ftrs,number_of_classes)
    weights = DenseNet121_Weights.DEFAULT  # or choose another variant if needed
    densenet_model = models.densenet121(weights=weights)


    # Replace the classifier
    num_ftrs = densenet_model.classifier.in_features
    densenet_model.classifier = nn.Linear(num_ftrs, num_classes)  # Adjust 'num_classes' to your dataset

    # Device Configuration and Model Transfer
    device = set_device()
    #densenet_model.to(device)
  
#device = set_device()
#resnet50_model = resnet50_model.to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.SGD(densenet_model.parameters(),lr=lr, momentum=momentum, weight_decay=weight_decay)
# Load checkpoint if exists
    checkpoint_path = "model_best.pth.tar"
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    densenet_model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
#mean = [0.4868, 0.4868, 0.4868]
#std = [0.1944, 0.1944, 0.1944]
    densenet_model.eval()



    image_transforms = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    ThresholdTransform(thr_255=75),
    #transforms.Normalize((0.5), (0.5))
    transforms.Normalize(torch.Tensor(mean), torch.Tensor(std))
])
    
    return densenet_model,image_transforms

densenet_model, image_transforms = load_model()
model = densenet_model.eval()
def set_device():
    if torch.cuda.is_available():
        print("Using CUDA GPU")
        dev = "cuda:0"
    elif torch.backends.mps.is_available():
     #   print("Using Mac GPU")
        dev = "mps"
    else:
        print("Using CPU")
        dev = "cpu"
    return torch.device(dev)
def classify(image_path):
    image = Image.open(image_path).convert('RGB')
    # Keep the original image for plotting
    original_image = image.copy()
    image = image_transforms(image).float()
    image = image.unsqueeze(0)
    output = densenet_model(image)
    _, predicted = torch.max(output.data, 1)
    # Return both the prediction and the original image
    #return classes[predicted.item()], original_image
    return classes[predicted.item()]

def main():
    for i in range(10):
        path = "/Users/ddayley/Desktop/gas/data/images_copy/" + str(i)
        files = os.listdir(path)
        for file in files:
            if file != ".DS_Store":
                guess, original_image = classify(os.path.join(path, file))  # Now returns original image as well
                if str(i) != guess:
                    print(str(i) + " " + file + " Debug-Predict: " + guess)
                    # Display the original image
                    plt.imshow(original_image)
                    plt.title("Failed Image: Expected {}, Predicted {}".format(i, guess))
                    plt.show()
if __name__ == '__main__':
    main()
