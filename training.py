"""
Training utilities and loss functions.
Implements the training loop with gradient descent.
"""

import numpy as np
from utils import cross_entropy_loss, one_hot_encode, accuracy


class Trainer:
    """
    Handles training of the neural network.
    """
    
    def __init__(self, neural_network, backpropagation, learning_rate=0.01):
        """
        Initialize trainer.
        
        Args:
            neural_network: NeuralNetwork instance
            backpropagation: Backpropagation instance
            learning_rate: Learning rate for gradient descent
        """
        self.nn = neural_network
        self.backprop = backpropagation
        self.learning_rate = learning_rate
        self.training_history = {
            'loss': [],
            'accuracy': []
        }
    
    def train_epoch(self, X_train, y_train, batch_size=32):
        """
        Train for one epoch (one pass through the entire dataset).
        
        Args:
            X_train: Training data
            y_train: Training labels (integer labels, not one-hot)
            batch_size: Size of each batch
            
        Returns:
            Average loss and accuracy for the epoch
        """
        losses = []
        accuracies = []
        
        # Convert labels to one-hot encoding
        y_train_one_hot = one_hot_encode(y_train, self.nn.layer_sizes[-1])
        
        # Train on batches
        for X_batch, y_batch in self._create_batches(X_train, y_train_one_hot, batch_size):
            # Forward pass
            y_pred = self.nn.forward(X_batch)
            
            # Compute loss
            loss = cross_entropy_loss(y_pred, y_batch)
            losses.append(loss)
            
            # Compute accuracy
            predictions = np.argmax(y_pred, axis=1)
            true_labels = np.argmax(y_batch, axis=1)
            acc = accuracy(predictions, true_labels)
            accuracies.append(acc)
            
            # Backward pass (compute gradients)
            gradients = self.backprop.compute_gradients(X_batch, y_batch)
            
            # Update weights
            self.backprop.update_weights(gradients, self.learning_rate)
        
        avg_loss = np.mean(losses)
        avg_accuracy = np.mean(accuracies)
        
        return avg_loss, avg_accuracy
    
    def train(self, X_train, y_train, X_val, y_val, epochs=10, batch_size=32, verbose=True):
        """
        Train the neural network for multiple epochs.
        
        Args:
            X_train: Training data
            y_train: Training labels
            X_val: Validation data
            y_val: Validation labels
            epochs: Number of training epochs
            batch_size: Size of each batch
            verbose: Whether to print training progress
            
        Returns:
            Training history dictionary
        """
        for epoch in range(epochs):
            # Train for one epoch
            train_loss, train_acc = self.train_epoch(X_train, y_train, batch_size)
            
            # Evaluate on validation set
            val_loss, val_acc = self.evaluate(X_val, y_val)
            
            # Store history
            self.training_history['loss'].append({
                'train': train_loss,
                'val': val_loss
            })
            self.training_history['accuracy'].append({
                'train': train_acc,
                'val': val_acc
            })
            
            if verbose:
                print(f"Epoch {epoch + 1}/{epochs}")
                print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
                print(f"  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
                print()
        
        return self.training_history
    
    def evaluate(self, X, y):
        """
        Evaluate the model on given data.
        
        Args:
            X: Input data
            y: True labels (integer labels)
            
        Returns:
            Loss and accuracy
        """
        # Convert labels to one-hot
        y_one_hot = one_hot_encode(y, self.nn.layer_sizes[-1])
        
        # Forward pass
        y_pred = self.nn.forward(X)
        
        # Compute loss
        loss = cross_entropy_loss(y_pred, y_one_hot)
        
        # Compute accuracy
        predictions = np.argmax(y_pred, axis=1)
        acc = accuracy(predictions, y)
        
        return loss, acc
    
    def _create_batches(self, X, y, batch_size):
        """
        Create batches from dataset.
        
        Args:
            X: Input data
            y: Labels (one-hot encoded)
            batch_size: Size of each batch
            
        Yields:
            Batches of (X_batch, y_batch)
        """
        num_samples = X.shape[0]
        indices = np.arange(num_samples)
        np.random.shuffle(indices)
        
        for i in range(0, num_samples, batch_size):
            batch_indices = indices[i:i + batch_size]
            yield X[batch_indices], y[batch_indices]

