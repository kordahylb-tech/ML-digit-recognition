# Handwritten Digit Recognition - From Scratch

A machine learning project that implements a neural network with backpropagation from scratch to recognize handwritten digits (0-9). This project is designed for learning purposes - all algorithms are implemented mathematically without using pre-built ML libraries.

## Project Structure

```
Test1/
├── README.md                 # This file
├── main.py                   # Main entry point
├── neural_network.py         # Neural network implementation
├── backpropagation.py        # Backpropagation algorithm
├── data_loader.py            # Data loading and preprocessing
├── training.py               # Training loop and loss functions
├── evaluation.py             # Model evaluation utilities
├── utils.py                  # Helper functions (activations, etc.)
└── requirements.txt          # Python dependencies
```

## Architecture Overview

### Neural Network
- **Input Layer**: 784 neurons (28x28 pixels for MNIST images)
- **Hidden Layer(s)**: Configurable (default: 128 neurons)
- **Output Layer**: 10 neurons (one for each digit 0-9)

### Key Components

1. **Neural Network (`neural_network.py`)**
   - Forward propagation
   - Weight initialization
   - Layer management

2. **Backpropagation (`backpropagation.py`)**
   - Gradient computation
   - Weight updates
   - Bias updates

3. **Activation Functions (`utils.py`)**
   - Sigmoid
   - ReLU
   - Softmax (for output layer)

4. **Loss Function (`training.py`)**
   - Cross-entropy loss
   - Mean squared error (optional)

5. **Data Loader (`data_loader.py`)**
   - MNIST dataset loading
   - Data normalization
   - Batch creation

## Mathematical Foundation

### Forward Propagation
For each layer:
- `z = W * x + b`
- `a = activation(z)`

### Backpropagation
1. Compute output error: `δ_output = (predicted - actual) * activation_derivative`
2. Propagate error backward through layers
3. Compute gradients: `∂L/∂W = δ * a_prev^T`
4. Update weights: `W = W - learning_rate * ∂L/∂W`

## Installation

1. Clone the repository:
```bash
git clone https://github.com/kordahylb-tech/ML-digit-recognition.git
cd ML-digit-recognition
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

Run training:
```bash
python main.py
```

The script will automatically download the MNIST dataset (~11.5 MB) on first run.

## Learning Objectives

- Understanding neural network architecture
- Implementing forward propagation
- Implementing backpropagation algorithm
- Understanding gradient descent
- Learning activation functions and their derivatives
- Understanding loss functions

