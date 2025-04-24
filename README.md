# Brain CT Anomaly Detection with Convolutional Autoencoders

This project implements a convolutional autoencoder in TensorFlow/Keras to distinguish between healthy and tumorous brain CT images.

## Overview
- Utilizes a convolutional autoencoder for anomaly detection
- Model is trained only on healthy brain CT images
- Threshold value is established by inference on healthy brain CT images
- Tumorous images are detected by high reconstruction error and surpassing the threshold-value

## Dataset

The dataset includes:
- Healthy brain CT scans
- Tumorous brain CT scans
- Healthy brain MRI scans
- Tumorous brain MRI scans

Link to dataset:
- [Kaggle] https://www.kaggle.com/datasets/murtozalikhon/brain-tumor-multimodal-image-ct-and-mri

