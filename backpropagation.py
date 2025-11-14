"""
Backpropagation algorithm implementation from scratch.
This module implements the backward pass to compute gradients.
"""

import numpy as np
from utils import sigmoid_derivative, relu_derivative, softmax


class Backpropagation:
    """
    Backpropagation algorithm for computing gradients.
    """
    
    def __init__(self, neural_network, loss_function='cross_entropy'):
        """
        Initialize backpropagation.
        
        Args:
            neural_network: NeuralNetwork instance
            loss_function: Type of loss function ('cross_entropy' or 'mse')
        """
        self.nn = neural_network
        self.loss_function = loss_function
    
    def compute_gradients(self, X, y_true):
        """
        Compute gradients using backpropagation algorithm.
        
        Mathematical steps:
        1. Forward pass (already done in neural_network.forward())
        2. Compute output error: δ^L = ∇_a C ⊙ σ'(z^L)
        3. Backpropagate error: δ^l = ((W^(l+1))^T * δ^(l+1)) ⊙ σ'(z^l)
        4. Compute gradients: ∂C/∂W^l = δ^l * (a^(l-1))^T
        5. Compute bias gradients: ∂C/∂b^l = δ^l
        
        Args:
            X: Input data (shape: [batch_size, input_size])
            y_true: True labels, one-hot encoded (shape: [batch_size, num_classes])
            
        Returns:
            Dictionary containing weight and bias gradients
        """
        # Forward pass (activations and z_values are stored in neural_network)
        y_pred = self.nn.forward(X)
        
        batch_size = X.shape[0]
        num_layers = self.nn.num_layers
        
        # Initialize gradient storage
        weight_gradients = [np.zeros_like(w) for w in self.nn.weights]
        bias_gradients = [np.zeros_like(b) for b in self.nn.biases]
        
        # Step 1: Compute output layer error
        # For cross-entropy with softmax: δ^L = y_pred - y_true
        if self.loss_function == 'cross_entropy':
            # With softmax and cross-entropy, the derivative simplifies
            delta = (y_pred - y_true).T  # Shape: [num_classes, batch_size]
        else:
            # For MSE: δ^L = (y_pred - y_true) * activation_derivative(z^L)
            # Since output uses softmax, we use the softmax derivative
            output_z = self.nn.z_values[-1]
            delta = (y_pred - y_true).T * softmax(output_z) * (1 - softmax(output_z))
        
        # Step 2: Compute gradients for output layer
        weight_gradients[-1] = np.dot(delta, self.nn.activations[-2].T) / batch_size
        bias_gradients[-1] = np.mean(delta, axis=1, keepdims=True)
        
        # Step 3: Backpropagate error through hidden layers
        for l in range(num_layers - 2, 0, -1):  # Go backwards through hidden layers
            # Compute error for layer l
            # δ^l = ((W^(l+1))^T * δ^(l+1)) ⊙ σ'(z^l)
            delta = np.dot(self.nn.weights[l].T, delta)
            
            # Apply activation derivative
            if self.nn.activation == 'relu':
                delta = delta * relu_derivative(self.nn.z_values[l-1])
            else:
                delta = delta * sigmoid_derivative(self.nn.z_values[l-1])
            
            # Compute gradients
            weight_gradients[l-1] = np.dot(delta, self.nn.activations[l-1].T) / batch_size
            bias_gradients[l-1] = np.mean(delta, axis=1, keepdims=True)
        
        return {
            'weight_gradients': weight_gradients,
            'bias_gradients': bias_gradients
        }
    
    def update_weights(self, gradients, learning_rate):
        """
        Update weights and biases using computed gradients.
        
        Weight update rule: W = W - learning_rate * ∂C/∂W
        Bias update rule: b = b - learning_rate * ∂C/∂b
        
        Args:
            gradients: Dictionary with 'weight_gradients' and 'bias_gradients'
            learning_rate: Learning rate for gradient descent
        """
        # Update weights
        for i in range(len(self.nn.weights)):
            self.nn.weights[i] -= learning_rate * gradients['weight_gradients'][i]
            self.nn.biases[i] -= learning_rate * gradients['bias_gradients'][i]

