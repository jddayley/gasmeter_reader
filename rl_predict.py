
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
import os
import torch
import torchvision.models as models
from torchvision.models import DenseNet121_Weights

class CustomCNN(nn.Module):
    def __init__(self):
        super(CustomCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=8, stride=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=1)

        # Recalculating flattened size
        size_after_conv1 = 217  # Output size after conv1
        size_after_conv2 = 214  # Output size after conv2

        flattened_size = 64 * size_after_conv2 * size_after_conv2
        self.fc1 = nn.Linear(flattened_size, 128)
        self.fc2 = nn.Linear(128, 2)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view(x.size(0), -1)  # Flatten
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Image transformation
def transform_image(image_path):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    image = Image.open(image_path).convert('RGB')  # Convert image to RGB
    image = transform(image).unsqueeze(0)  # Add batch dimension
    return image

# Function to predict the class of an image
def predict_image(model, image_path):
    image = transform_image(image_path)
    with torch.no_grad():
        outputs = model(image)
        _, predicted = torch.max(outputs, 1)
        return predicted.item()

# Load the model
#model = load_custom_model('best_checkpoint.pth.tar')




# Predict the class of an image
# image_path = 'path_to_your_image.jpg'  # Replace with your image path
# predicted_class = predict_image(model, image_path)
# print(f"The predicted class for the image is: {predicted_class}")
def main():
    model = load_custom_model('best_checkpoint.pth.tar')  # Load the model
    
    for i in range(10):
        path = "/Users/ddayley/Desktop/gas/data/images_copy/" + str(i)
        files = os.listdir(path)
        for file in files:
            if file != ".DS_Store":
                full_path = os.path.join(path, file)
                guess = predict_image(model, full_path)  # Corrected order of arguments
                if str(i) != str(guess):
                    print(f"{i} {file} Debug-Predict: {guess}")
def find_flattened_size():
    # Create a dummy input tensor of the correct input size
    dummy_input = torch.randn(1, 3, 224, 224)  # Assuming input size is 224x224x3

    # Initialize the model
    model = CustomCNN()

    # Forward pass of the dummy input through the conv layers only
    with torch.no_grad():
        x = F.relu(model.conv1(dummy_input))
        x = F.relu(model.conv2(x))
        flattened_size = x.view(x.size(0), -1).shape[1]
    
    return flattened_size

def load_custom_model(start_from_scratch=True, checkpoint_path='best_checkpoint.pth.tar'):
    # Choose whether to start from scratch or load the checkpoint
    if start_from_scratch:
        # Start training from scratch
        model = models.densenet121(weights=DenseNet121_Weights.DEFAULT)
        num_ftrs = model.classifier.in_features
        model.classifier = nn.Linear(num_ftrs, 10)  # Adjust the number of classes if different
    else:
        # Load the existing checkpoint
        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path)
            model = models.densenet121(weights=DenseNet121_Weights.DEFAULT)
            model.load_state_dict(checkpoint['state_dict'])
        else:
            # Handle the case where the checkpoint file doesn't exist
            print(f"Checkpoint file '{checkpoint_path}' not found. Starting from scratch.")
            model = models.densenet121(weights=DenseNet121_Weights.DEFAULT)
            num_ftrs = model.classifier.in_features
            model.classifier = nn.Linear(num_ftrs, 10)  # Adjust the number of classes if different

    model.eval()
    return model

# Usage


if __name__ == '__main__':
    main()

# Load the original DenseNet model
    # model = models.densenet121(pretrained=False)

    # # Load your checkpoint
    # checkpoint = torch.load('best_checkpoint.pth.tar')
    # model.load_state_dict(checkpoint['state_dict'])
    # checkpoint = torch.load('best_checkpoint.pth.tar')
    # # print(checkpoint.keys())
    # # print(checkpoint['state_dict'].keys())
    # checkpoint = torch.load('best_checkpoint.pth.tar', map_location=torch.device('cpu'))
    # print(checkpoint['state_dict'].keys())

# Call the function
    # flattened_size = find_flattened_size()
    # print("Calculated Flattened Size:", flattened_size)
