"""
Neural Network implementation from scratch.
This module implements a multi-layer perceptron with forward propagation.
"""

import numpy as np
from utils import sigmoid, relu, softmax


class NeuralNetwork:
    """
    A multi-layer neural network implemented from scratch.
    """
    
    def __init__(self, layer_sizes, activation='sigmoid', random_seed=42):
        """
        Initialize the neural network.
        
        Args:
            layer_sizes: List of integers representing neurons in each layer
                        [input_size, hidden1, hidden2, ..., output_size]
            activation: Activation function for hidden layers ('sigmoid' or 'relu')
            random_seed: Random seed for reproducibility
        """
        self.layer_sizes = layer_sizes
        self.activation = activation
        self.num_layers = len(layer_sizes)
        
        # Set random seed for reproducibility
        np.random.seed(random_seed)
        
        # Initialize weights and biases
        self.weights = []
        self.biases = []
        
        # Initialize weights using He initialization for ReLU, Xavier for sigmoid
        for i in range(self.num_layers - 1):
            if activation == 'relu':
                # He initialization: weights ~ N(0, sqrt(2/n))
                weight = np.random.randn(layer_sizes[i+1], layer_sizes[i]) * np.sqrt(2.0 / layer_sizes[i])
            else:
                # Xavier initialization: weights ~ N(0, sqrt(1/n))
                weight = np.random.randn(layer_sizes[i+1], layer_sizes[i]) * np.sqrt(1.0 / layer_sizes[i])
            
            bias = np.zeros((layer_sizes[i+1], 1))
            
            self.weights.append(weight)
            self.biases.append(bias)
        
        # Store activations and z values for backpropagation
        self.activations = []
        self.z_values = []
    
    def forward(self, X):
        """
        Forward propagation through the network.
        
        Args:
            X: Input data (shape: [batch_size, input_size])
            
        Returns:
            Output predictions (shape: [batch_size, output_size])
        """
        # Reset storage for this forward pass
        self.activations = [X.T]  # Store as column vectors
        self.z_values = []
        
        # Forward pass through hidden layers
        for i in range(self.num_layers - 2):  # All layers except output
            # Linear transformation: z = W * a + b
            z = np.dot(self.weights[i], self.activations[i]) + self.biases[i]
            self.z_values.append(z)
            
            # Apply activation function
            if self.activation == 'relu':
                a = relu(z)
            else:
                a = sigmoid(z)
            
            self.activations.append(a)
        
        # Output layer (no activation here, will apply softmax in loss calculation)
        z_output = np.dot(self.weights[-1], self.activations[-1]) + self.biases[-1]
        self.z_values.append(z_output)
        
        # Apply softmax for output layer
        a_output = softmax(z_output.T).T  # Transpose for consistency
        self.activations.append(a_output)
        
        return a_output.T  # Return as row vectors [batch_size, output_size]
    
    def predict(self, X):
        """
        Make predictions on input data.
        
        Args:
            X: Input data (shape: [batch_size, input_size])
            
        Returns:
            Predicted class indices
        """
        predictions = self.forward(X)
        return np.argmax(predictions, axis=1)
    
    def get_weights(self):
        """Return all weights."""
        return self.weights
    
    def get_biases(self):
        """Return all biases."""
        return self.biases
    
    def set_weights(self, weights):
        """Set all weights."""
        self.weights = weights
    
    def set_biases(self, biases):
        """Set all biases."""
        self.biases = biases

