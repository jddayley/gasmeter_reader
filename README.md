# Gas Meter Reader

## Background
This project is an enhanced adaptation of David Padbury's Power Meter Reader, specifically designed to read gas meters for Eversource in Hopkinton, MA. Initially using HoughCircles for feature detection, the project has evolved to incorporate a machine learning model, significantly improving accuracy. The process involves cropping to the meter's dials, normalization, and advanced image processing techniques.

## Files and Their Functionalities

### image_class_gpt.py
- **Purpose**: Image classification.
- **Features**: Configures the device for image processing using torchvision, torch, and transforms.

### image_predict.py
- **Purpose**: Manages image classification predictions.
- **Features**: Utilizes torchvision, torch, and PIL for image classification tasks.

### gasmeter.py
- **Purpose**: Reads the gas meter and publishes readings via MQTT.
- **Features**: Integrates with data publishing systems, employing sys, os, json, etc.

### gas_meter_reader.py
- **Purpose**: Primary script for reading the gas meter using machine vision.
- **Features**: Implements image processing and machine vision techniques.

### globals.py
- **Purpose**: Manages global variables of the project.
- **Features**: Defines variables like flags, error thresholds, sleep times, and dataset paths.

### threshtransform.py
- **Purpose**: Applies a threshold transform to images.
- **Features**: Includes the `ThresholdTransform` class for image thresholding.

## Usage

### Training the Model
- **Setting Up Global Variables**: Configure parameters in `globals.py`, including the training dataset path, error thresholds, etc., before training.
- **Executing Training**: Utilize `image_class_gpt.py` and `image_predict.py` for training. The training utilizes images from the dataset path in `globals.py`, ensuring they represent the meter readings accurately.
- **Model Training Considerations**: The model learns to recognize the gas meter's specific characteristics from the provided images. It's crucial for the images to encompass various readings and lighting conditions for robustness.

### Deployment
Deploy the trained model using `gasmeter.py` and `gas_meter_reader.py` for actual meter reading and data publishing. The trained model interprets the gas meter readings and, if configured, publishes them using MQTT.

## Hardware
- **Webcam**: Using a Wyze v3 USB webcam, modified for close focus based on a YouTube tutorial ([link](https://www.youtube.com/watch?v=PnqDFVH_lfU&t=367s)).
- **Lighting**: Employing LED lighting ([Amazon link](https://www.amazon.com/gp/product/B072QWJRBS/ref=ppx_yo_dt_b_asin_title_o02_s00?ie=UTF8&th=1)).
- **Enclosure**: Housed in a Register Duck Boot ([Lowes link](https://www.lowes.com/pd/IMPERIAL-10-in-x-6-in-x-6-in-Galvanized-Steel-Straight-Register-Duct-Boot/1000237469)).
- **Custom 3D Printed Enclosure**: An enclosure created with a 3D printer. STL file to be included.

## Data Errors and Solutions
- **Assumption of Increasing Readings**: The system assumes a continual increase in gas readings.
- **Limitation on Reading Increase**: Presumes that readings should not increase by more than 3 units since the last value.
- **Correction Method**: Corrects discrepancies by replacing the first digit with the first digit from the previous reading and reassessing. This process is repeated for the second and third digits as needed.

## Next Steps
- **Creating a Controlled Environment**: Aiming to enhance reliability by constructing a controlled environment using a metal enclosure and LED lighting, which should mitigate data errors significantly.
