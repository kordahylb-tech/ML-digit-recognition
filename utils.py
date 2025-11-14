"""
Utility functions for activation functions and their derivatives.
All functions are implemented from scratch using only numpy.
"""

import numpy as np


def sigmoid(x):
    """
    Sigmoid activation function: σ(x) = 1 / (1 + e^(-x))
    
    Args:
        x: Input array
        
    Returns:
        Sigmoid of input
    """
    # Clip to prevent overflow
    x = np.clip(x, -500, 500)
    return 1 / (1 + np.exp(-x))


def sigmoid_derivative(x):
    """
    Derivative of sigmoid: σ'(x) = σ(x) * (1 - σ(x))
    
    Args:
        x: Input array (can be sigmoid output or raw input)
        
    Returns:
        Derivative of sigmoid
    """
    s = sigmoid(x)
    return s * (1 - s)


def relu(x):
    """
    ReLU activation function: ReLU(x) = max(0, x)
    
    Args:
        x: Input array
        
    Returns:
        ReLU of input
    """
    return np.maximum(0, x)


def relu_derivative(x):
    """
    Derivative of ReLU: ReLU'(x) = 1 if x > 0, else 0
    
    Args:
        x: Input array
        
    Returns:
        Derivative of ReLU
    """
    return (x > 0).astype(float)


def softmax(x):
    """
    Softmax activation function: softmax(x_i) = e^(x_i) / Σ(e^(x_j))
    
    Args:
        x: Input array (shape: [batch_size, num_classes])
        
    Returns:
        Softmax probabilities
    """
    # Subtract max for numerical stability
    exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)


def one_hot_encode(labels, num_classes):
    """
    Convert integer labels to one-hot encoded vectors.
    
    Args:
        labels: Array of integer labels
        num_classes: Number of classes
        
    Returns:
        One-hot encoded array
    """
    one_hot = np.zeros((len(labels), num_classes))
    one_hot[np.arange(len(labels)), labels] = 1
    return one_hot


def cross_entropy_loss(y_pred, y_true):
    """
    Cross-entropy loss: L = -Σ(y_true * log(y_pred))
    
    Args:
        y_pred: Predicted probabilities (after softmax)
        y_true: True labels (one-hot encoded)
        
    Returns:
        Cross-entropy loss value
    """
    # Add small epsilon to prevent log(0)
    epsilon = 1e-15
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    return -np.mean(np.sum(y_true * np.log(y_pred), axis=1))


def accuracy(y_pred, y_true):
    """
    Calculate accuracy: percentage of correct predictions.
    
    Args:
        y_pred: Predicted class indices
        y_true: True class indices
        
    Returns:
        Accuracy as a float between 0 and 1
    """
    return np.mean(y_pred == y_true)

