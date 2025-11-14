"""
Data loading and preprocessing utilities.
Handles MNIST dataset loading and preparation.
"""

import numpy as np
import urllib.request
import urllib.error
import ssl
import gzip
import os

# Create SSL context that doesn't verify certificates (for compatibility)
ssl_context = ssl.create_default_context()
ssl_context.check_hostname = False
ssl_context.verify_mode = ssl.CERT_NONE


def download_mnist():
    """
    Download MNIST dataset files if they don't exist.
    MNIST dataset is publicly available and commonly used for learning.
    """
    files = {
        'train_images': 'train-images-idx3-ubyte.gz',
        'train_labels': 'train-labels-idx1-ubyte.gz',
        'test_images': 't10k-images-idx3-ubyte.gz',
        'test_labels': 't10k-labels-idx1-ubyte.gz'
    }
    
    data_dir = 'data'
    os.makedirs(data_dir, exist_ok=True)
    
    for key, filename in files.items():
        filepath = os.path.join(data_dir, filename)
        if not os.path.exists(filepath):
            print(f"Downloading {filename}...")
            downloaded = False
            
            # Try multiple reliable sources
            urls_to_try = [
                (f'https://github.com/pjreddie/mnist/raw/master/{filename}', 'GitHub mirror'),
                (f'http://yann.lecun.com/exdb/mnist/{filename}', 'Original LeCun site'),
                (f'https://storage.googleapis.com/cvdf-datasets/mnist/{filename}', 'Google storage'),
            ]
            
            for url, source_name in urls_to_try:
                try:
                    print(f"  Trying {source_name}...")
                    # Create request with timeout
                    req = urllib.request.Request(url)
                    req.add_header('User-Agent', 'Mozilla/5.0')
                    # Only use SSL context for HTTPS URLs
                    if url.startswith('https'):
                        response = urllib.request.urlopen(req, timeout=30, context=ssl_context)
                    else:
                        response = urllib.request.urlopen(req, timeout=30)
                    with response:
                        with open(filepath, 'wb') as out_file:
                            out_file.write(response.read())
                    print(f"  Successfully downloaded from {source_name}!")
                    downloaded = True
                    break
                except Exception as e:
                    print(f"  {source_name} failed: {str(e)[:100]}")
                    continue
            
            if not downloaded:
                raise ConnectionError(
                    f"Failed to download {filename}. "
                    f"Please check your internet connection or manually download from:\n"
                    f"  https://github.com/pjreddie/mnist/raw/master/{filename}\n"
                    f"  and place it in the 'data' folder."
                )
    
    return data_dir


def load_mnist_images(filename):
    """
    Load MNIST images from IDX file format.
    
    Args:
        filename: Path to the image file
        
    Returns:
        Images as numpy array (shape: [num_images, 28, 28])
    """
    with gzip.open(filename, 'rb') as f:
        # Skip header (16 bytes for images)
        f.read(16)
        # Read image data
        buf = f.read()
        data = np.frombuffer(buf, dtype=np.uint8)
        # Reshape to [num_images, 28, 28]
        data = data.reshape(-1, 28, 28)
        return data


def load_mnist_labels(filename):
    """
    Load MNIST labels from IDX file format.
    
    Args:
        filename: Path to the label file
        
    Returns:
        Labels as numpy array
    """
    with gzip.open(filename, 'rb') as f:
        # Skip header (8 bytes for labels)
        f.read(8)
        # Read label data
        buf = f.read()
        labels = np.frombuffer(buf, dtype=np.uint8)
        return labels


def load_mnist_data():
    """
    Load and preprocess MNIST dataset.
    
    Returns:
        Tuple of (X_train, y_train, X_test, y_test)
    """
    # Download if needed
    data_dir = download_mnist()
    
    # Load training data
    train_images = load_mnist_images(os.path.join(data_dir, 'train-images-idx3-ubyte.gz'))
    train_labels = load_mnist_labels(os.path.join(data_dir, 'train-labels-idx1-ubyte.gz'))
    
    # Load test data
    test_images = load_mnist_images(os.path.join(data_dir, 't10k-images-idx3-ubyte.gz'))
    test_labels = load_mnist_labels(os.path.join(data_dir, 't10k-labels-idx1-ubyte.gz'))
    
    # Flatten images: [num_images, 28, 28] -> [num_images, 784]
    X_train = train_images.reshape(train_images.shape[0], -1)
    X_test = test_images.reshape(test_images.shape[0], -1)
    
    # Normalize pixel values to [0, 1]
    X_train = X_train.astype(np.float32) / 255.0
    X_test = X_test.astype(np.float32) / 255.0
    
    return X_train, train_labels, X_test, test_labels


def create_batches(X, y, batch_size):
    """
    Create batches from dataset.
    
    Args:
        X: Input data
        y: Labels
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

