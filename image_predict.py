import torch
import torchvision
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
import PIL.Image as Image
import os
import sys
from threshtransform import ThresholdTransform  # Ensure this is correctly implemented
import torchvision.models as models
import torch.nn as nn
import matplotlib.pyplot as plt

# Calculate mean and standard deviation
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
    return mean.numpy(), std.numpy()  # Convert to numpy array for later use in Normalize

# Configuration
config = {
    "checkpoint_path": "checkpoints/checkpoint.pth.tar",
    "train_dataset_path": "data/images_copy",
    "num_classes": 10,
    "device": "auto",
    "batch_size": 32,
    "classes": [str(i) for i in range(10)],
}

# Set device based on availability and preference
def set_device():
    if torch.cuda.is_available():
        print("Using CUDA GPU")
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        print("Using Mac GPU")
        return torch.device("mps")
    else:
        print("Using CPU")
        return torch.device("cpu")

# Load the model from checkpoint
def load_model(checkpoint_path, device):
    model = models.densenet121(pretrained=False)  # Initialize model
    num_ftrs = model.classifier.in_features
    model.classifier = nn.Linear(num_ftrs, config["num_classes"])
    model.to(device)

    if os.path.isfile(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        if 'model_state' in checkpoint:
            model.load_state_dict(checkpoint['model_state'])
        elif 'state_dict' in checkpoint:
            model.load_state_dict(checkpoint['state_dict'])
        else:
            model.load_state_dict(checkpoint)
        print("Model loaded successfully from checkpoint.")
    else:
        print(f"Checkpoint not found at {checkpoint_path}", file=sys.stderr)
        sys.exit(1)

    model.eval()
    return model

# Update this function to accept mean and std as parameters
def get_transforms(mean, std):
    return transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        ThresholdTransform(thr_255=75),
        transforms.Normalize(mean=mean, std=std)
    ])
def gas_classify(image_path):
    # Set up device
    device = set_device()
    
    # Load the model
    model = load_model(config["checkpoint_path"], device)
    
    # Prepare the loader for mean/std calculation (if not already calculated)
    global mean, std  # Use global variables to avoid recalculating
    try:
        _ = mean + std  # Try to use existing mean and std
    except NameError:  # Calculate if not already done
        initial_transforms = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])
        initial_dataset = datasets.ImageFolder(root=config["train_dataset_path"], transform=initial_transforms)
        initial_loader = DataLoader(initial_dataset, batch_size=config["batch_size"], shuffle=False)
        mean, std = calculate_mean_std(initial_loader)
    
    # Get image transforms using the calculated mean and std
    image_transforms = get_transforms(mean, std)
    
    # Classify the image
    predicted_class, original_image = classify(model, image_path, image_transforms, device)  
    return predicted_class, original_image

# Classify a given image
def classify(model, image_path, transforms, device):
    image = Image.open(image_path).convert('RGB')
    transformed_image = transforms(image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        output = model(transformed_image)
        _, predicted = torch.max(output, 1)
        predicted_class = config["classes"][predicted.item()]
    
    return predicted_class, image  # Return PIL image for plotting

def main():
    device = set_device()
    
    # Prepare initial loader for mean/std calculation
    initial_transforms = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])
    initial_dataset = datasets.ImageFolder(root=config["train_dataset_path"], transform=initial_transforms)
    initial_loader = DataLoader(initial_dataset, batch_size=config["batch_size"], shuffle=False)
    mean, std = calculate_mean_std(initial_loader)
    
    model = load_model(config["checkpoint_path"], device)
    image_transforms = get_transforms(mean, std)
    
    test_directory = config["train_dataset_path"]
    for i in range(len(config["classes"])):
        path = os.path.join(test_directory, str(i))
        if not os.path.isdir(path):
            continue
        files = [f for f in os.listdir(path) if not f.startswith('.')]
        for file in files:
            image_path = os.path.join(path, file)
            predicted_class, original_image = classify(model, image_path, image_transforms, device)
            if str(i) != predicted_class:
                print(f"Image: {file}, Predicted class: {predicted_class}")
                plt.imshow(original_image)
                plt.title(f"Class: {i} - Predicted: {predicted_class}")
                plt.show()

if __name__ == '__main__':
    main()
