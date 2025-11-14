"""
Main entry point for the handwritten digit recognition project.
This script orchestrates the entire training and evaluation process.
"""

import numpy as np
from neural_network import NeuralNetwork
from backpropagation import Backpropagation
from data_loader import load_mnist_data
from training import Trainer
from evaluation import plot_training_history, visualize_predictions, confusion_matrix, plot_confusion_matrix


def main():
    """
    Main function to train and evaluate the neural network.
    """
    print("=" * 60)
    print("Handwritten Digit Recognition - From Scratch")
    print("=" * 60)
    print()
    
    # Load data
    print("Loading MNIST dataset...")
    X_train, y_train, X_test, y_test = load_mnist_data()
    print(f"Training samples: {X_train.shape[0]}")
    print(f"Test samples: {X_test.shape[0]}")
    print(f"Image shape: {X_train.shape[1]} (28x28 flattened)")
    print()
    
    # Split training data into train and validation sets
    val_size = int(0.1 * len(X_train))
    X_val = X_train[:val_size]
    y_val = y_train[:val_size]
    X_train = X_train[val_size:]
    y_train = y_train[val_size:]
    
    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Validation set: {X_val.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")
    print()
    
    # Define network architecture
    # Input: 784 (28x28 pixels)
    # Hidden: 128 neurons
    # Output: 10 (digits 0-9)
    layer_sizes = [784, 128, 10]
    
    print("Initializing neural network...")
    print(f"Architecture: {layer_sizes}")
    total_params = sum(layer_sizes[i] * layer_sizes[i+1] + layer_sizes[i+1] for i in range(len(layer_sizes)-1))
    print(f"Total parameters: {total_params}")
    print()
    
    # Create neural network
    nn = NeuralNetwork(layer_sizes, activation='sigmoid', random_seed=42)
    
    # Create backpropagation
    backprop = Backpropagation(nn, loss_function='cross_entropy')
    
    # Create trainer
    trainer = Trainer(nn, backprop, learning_rate=0.01)
    
    # Train the model
    print("Starting training...")
    print("-" * 60)
    history = trainer.train(
        X_train, y_train,
        X_val, y_val,
        epochs=10,
        batch_size=32,
        verbose=True
    )
    
    # Evaluate on test set
    print("Evaluating on test set...")
    test_loss, test_acc = trainer.evaluate(X_test, y_test)
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_acc:.4f} ({test_acc * 100:.2f}%)")
    print()
    
    # Visualizations
    print("Generating visualizations...")
    plot_training_history(history)
    visualize_predictions(nn, X_test, y_test, num_samples=10)
    
    # Confusion matrix
    predictions = nn.predict(X_test)
    cm = confusion_matrix(y_test, predictions)
    plot_confusion_matrix(cm)
    
    print()
    print("=" * 60)
    print("Training complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()

