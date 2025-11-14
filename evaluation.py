"""
Evaluation and visualization utilities.
"""

import numpy as np
import matplotlib.pyplot as plt
from utils import accuracy


def plot_training_history(history):
    """
    Plot training history (loss and accuracy over epochs).
    
    Args:
        history: Training history dictionary from Trainer
    """
    epochs = range(1, len(history['loss']) + 1)
    
    # Plot loss
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    train_losses = [h['train'] for h in history['loss']]
    val_losses = [h['val'] for h in history['loss']]
    plt.plot(epochs, train_losses, 'b-', label='Training Loss')
    plt.plot(epochs, val_losses, 'r-', label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)
    
    # Plot accuracy
    plt.subplot(1, 2, 2)
    train_accs = [h['train'] for h in history['accuracy']]
    val_accs = [h['val'] for h in history['accuracy']]
    plt.plot(epochs, train_accs, 'b-', label='Training Accuracy')
    plt.plot(epochs, val_accs, 'r-', label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('training_history.png')
    print("Training history saved to 'training_history.png'")
    plt.show()


def visualize_predictions(model, X_test, y_test, num_samples=10):
    """
    Visualize model predictions on test samples.
    
    Args:
        model: Trained NeuralNetwork instance
        X_test: Test images
        y_test: True labels
        num_samples: Number of samples to visualize
    """
    # Get random samples
    indices = np.random.choice(len(X_test), num_samples, replace=False)
    
    # Make predictions
    predictions = model.predict(X_test[indices])
    
    # Plot
    fig, axes = plt.subplots(2, 5, figsize=(12, 6))
    axes = axes.flatten()
    
    for i, idx in enumerate(indices):
        # Reshape image back to 28x28
        image = X_test[idx].reshape(28, 28)
        
        axes[i].imshow(image, cmap='gray')
        axes[i].set_title(f'True: {y_test[idx]}, Pred: {predictions[i]}')
        axes[i].axis('off')
        
        # Color title based on correctness
        if predictions[i] == y_test[idx]:
            axes[i].title.set_color('green')
        else:
            axes[i].title.set_color('red')
    
    plt.tight_layout()
    plt.savefig('predictions.png')
    print("Predictions visualization saved to 'predictions.png'")
    plt.show()


def confusion_matrix(y_true, y_pred, num_classes=10):
    """
    Compute confusion matrix.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        num_classes: Number of classes
        
    Returns:
        Confusion matrix as numpy array
    """
    cm = np.zeros((num_classes, num_classes), dtype=int)
    
    for i in range(len(y_true)):
        cm[y_true[i], y_pred[i]] += 1
    
    return cm


def plot_confusion_matrix(cm, class_names=None):
    """
    Plot confusion matrix.
    
    Args:
        cm: Confusion matrix
        class_names: List of class names (default: 0-9)
    """
    if class_names is None:
        class_names = [str(i) for i in range(10)]
    
    plt.figure(figsize=(10, 8))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names)
    plt.yticks(tick_marks, class_names)
    
    # Add text annotations
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], 'd'),
                    horizontalalignment="center",
                    color="white" if cm[i, j] > thresh else "black")
    
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png')
    print("Confusion matrix saved to 'confusion_matrix.png'")
    plt.show()

