# Traffic-sign-classifier

## Overview

Traffic signs provide important information to drivers, such as speed limits, warnings, and road regulations. Missing these signs can lead to accidents or traffic violations. The goal of this project is to build a machine learning model that can automatically recognize and classify traffic signs from images.

Using deep learning and computer vision techniques, the system can identify different categories of traffic signs and predict their labels with high accuracy. Such systems are commonly used in driver assistance technologies and autonomous vehicles.

## Problem Statement

Drivers need to continuously monitor road signs while driving, which can sometimes be challenging due to distractions, poor visibility, or unfamiliar roads. An automated traffic sign classification system can help by recognizing signs quickly and accurately.

This project aims to develop a model that can classify traffic sign images into their respective categories and assist in improving road safety.

## Features

* Recognizes and classifies multiple traffic sign categories.
* Uses deep learning for image-based classification.
* Automatically learns important visual features from training data.
* Provides predictions for unseen traffic sign images.
* Handles variations in sign appearance, lighting, and viewing angles.
* Can be integrated with larger driver assistance systems.

## Technologies Used

* Python
* TensorFlow / PyTorch
* OpenCV
* NumPy
* Pandas
* Matplotlib

## Workflow

1. Collect and prepare traffic sign images for training.
2. Resize and normalize the images.
3. Train a Convolutional Neural Network (CNN) on labeled data.
4. Extract features automatically through the network.
5. Classify new traffic sign images.
6. Display the predicted traffic sign category.

## Results

The trained model was able to correctly classify traffic signs across multiple categories with good accuracy. Testing showed that the model could effectively learn distinguishing features from the dataset and make reliable predictions on unseen images.

## Future Improvements

* Support real-time traffic sign recognition using live camera feeds.
* Improve performance under challenging weather and lighting conditions.
* Deploy the model on embedded and edge devices.
* Combine detection and classification into a single pipeline.
* Train on larger and more diverse traffic sign datasets.

## Conclusion

This project demonstrates how deep learning can be applied to solve real-world image classification problems. By automatically recognizing traffic signs, the system can contribute to safer driving and serve as a building block for advanced driver assistance and autonomous driving applications.
