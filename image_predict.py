import torchvision
import torch
import torchvision.transforms as transforms
import PIL.Image as Image
import cv2 
import os
import sys
path = "/Users/ddayley/Desktop/gas/output/"

def classify(image_path):
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
    model = torch.load('best_model.pth')
    mean = [0.4852, 0.4852, 0.4852]
    std = [0.1888, 0.1888, 0.1888]
    image_transforms = transforms.Compose([
       # transforms.Grayscale(num_output_channels=3),
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        #transforms.Normalize((0.5), (0.5))
        transforms.Normalize(torch.Tensor(mean), torch.Tensor(std))
    

    ])
    model = model.eval()
    image = Image.open(image_path).convert('RGB')
    image = image_transforms(image).float()
   # print (image.size())
    image = image.unsqueeze(0)
   # print (image.size())
    output = model(image)
   # print (output)
    max_elements, predicated = torch.max(output.data, 1)
    #print(max_elements)
    #print("Debug-Predict: " + classes[predicated.item()] )
   
    return str(classes[predicated.item()])

def main():
    files = os.listdir(path)
    for file in files:
        print(file)
        if file != ".DS_Store":
            classify( path + file )
if __name__ == '__main__':
    main()