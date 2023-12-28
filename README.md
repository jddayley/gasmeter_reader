# Gas Meter Reader

Background

This project adapts David Padbury's Power Meter Reader to read gas meters, specifically targeting Eversource in Hopkinton, MA. It involves cropping to the meter's dials, normalization, and feature detection using HoughCircles.

Files and Their Functionalities

# image_class_gpt.py
Used for image classification.
Sets up the device for image processing with torchvision, torch, and transforms.
# image_predict.py
Handles the prediction for image classification.
Utilizes torchvision, torch, and PIL for image classification.
# gasmeter.py
Reads the gas meter and publishes readings using MQTT.
Integrates with systems for data publishing using sys, os, json, etc.
# gas_meter_reader.py
Main script for reading the gas meter using machine vision.
Employs image processing and machine vision techniques.
# globals.py
Manages global variables for the project.
Defines variables like flags, error thresholds, sleep times, and dataset paths.
# threshtransform.py
Implements a threshold transform for images.
Contains ThresholdTransform class for image thresholding operations.
Usage

# Training the Model
Set Up Global Variables: Before training, configure the necessary parameters in globals.py. This includes the path to your training dataset, error thresholds, and other settings crucial for training.
# Training the Model: Use the image_class_gpt.py and image_predict.py scripts to train the model. The training process involves using images from the specified dataset path in globals.py. Ensure that the images used for training accurately represent the meter readings you intend to recognize.
Model Training Considerations: During training, the model will learn to recognize the specific characteristics of your gas meter based on the provided images. It's important that the images cover a range of different readings and lighting conditions to ensure robustness.
# Deployment
After training, deploy the model using gasmeter.py and gas_meter_reader.py for actual meter reading and data publishing. The system will use the trained model to interpret gas meter readings and, if configured, publish them using MQTT.

# Hardware

- I am using a Wyze v3 USB webcam.   It would not focus close enough so I followed the youtube instructions that allowed me  (https://www.youtube.com/watch?v=PnqDFVH_lfU&t=367s) to adjust the lense and focus on the gas meter.
- For lighting, I'm using the LED lighting. https://www.amazon.com/gp/product/B072QWJRBS/ref=ppx_yo_dt_b_asin_title_o02_s00?ie=UTF8&th=1
- I've enclosed everything in a Register Duck Boot. https://www.lowes.com/pd/IMPERIAL-10-in-x-6-in-x-6-in-Galvanized-Steel-Straight-Register-Duct-Boot/1000237469

# data errors
Lighting impacts the ability to take consistent measurements.  To combat it:
1. Take a number of consecutive frames prior to analysis (5)
2. Compute the median of the x, y, and radius values of detected circles
3. Use that center & radius for analysis of all frames
4. Ensure the value is always increasing.
5. Remove the first and/or second digital and compare last numbers.
6. Ensure the value has not increase too much.    

# Next steps
I am working on a controlled environment.  I purchased a metal enclosure and led lighting.  This will elimate the data errors and improve the reliability. 
