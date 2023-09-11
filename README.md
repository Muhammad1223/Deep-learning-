
# LeNet-5 Implementation on CIFAR-10

This repository contains the implementation of the classic LeNet-5 convolutional neural network, applied to the MNIST dataset. The code demonstrates loading the dataset, defining the network architecture, training the model, and evaluating its performance. Furthermore, it showcases visualizations for both CIFAR-10 and MNIST datasets.

## Features

1. **Dataset Visualization**: Visualizes the first few images from the CIFAR-10 dataset.
2. **Data Loading and Preprocessing**: Utilizes `torchvision` for loading MNIST and applying necessary transformations.
3. **LeNet-5 Architecture**: Implements the classic LeNet-5 architecture which comprises convolutional and fully connected layers.
4. **Training**: Trains the LeNet-5 model on MNIST dataset using the Adam optimizer and CrossEntropy loss.
5. **Evaluation**: Computes and displays the accuracy of the model after each epoch.
6. **Model Summary**: Provides a comprehensive summary of the model architecture, especially useful for understanding the dimensionality at each layer.
7. **Loss Visualization**: Plots the training and testing loss over epochs to visualize the performance of the model.

## Dependencies

- PyTorch
- torchvision
- torchsummary
- matplotlib

## Results

The notebook will display:

- First few images from the CIFAR-10 dataset.
- A summary of the LeNet-5 model architecture.
- Training and testing loss over epochs.


