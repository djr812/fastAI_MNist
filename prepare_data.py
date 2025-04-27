import struct
import numpy as np
from pathlib import Path
import os

def read_idx(filename):
    with open(filename, 'rb') as f:
        zero, data_type, dims = struct.unpack('>HBB', f.read(4))
        shape = tuple(struct.unpack('>I', f.read(4))[0] for d in range(dims))
        return np.frombuffer(f.read(), dtype=np.uint8).reshape(shape)

def prepare_mnist_data():
    # Create directory structure
    data_dir = Path('../mnist_data')
    if not data_dir.exists():
        data_dir.mkdir(parents=True)
    
    # Create subdirectories for each digit
    for i in range(10):
        digit_dir = data_dir / str(i)
        if not digit_dir.exists():
            digit_dir.mkdir(parents=True)
    
    # Read training data
    try:
        train_images = read_idx('../mnist_data/train-images.idx3-ubyte')
        train_labels = read_idx('../mnist_data/train-labels.idx1-ubyte')
        test_images = read_idx('../mnist_data/t10k-images.idx3-ubyte')
        test_labels = read_idx('../mnist_data/t10k-labels.idx1-ubyte')
    except FileNotFoundError as e:
        print("Error: MNIST dataset files not found!")
        print("Please ensure you have the following files in ../mnist_data:")
        print("- train-images.idx3-ubyte")
        print("- train-labels.idx1-ubyte")
        print("- t10k-images.idx3-ubyte")
        print("- t10k-labels.idx1-ubyte")
        print(f"\nSpecific error: {e}")
        return
    
    print("Processing training data...")
    from PIL import Image
    
    # Save training images
    for idx, (image, label) in enumerate(zip(train_images, train_labels)):
        # Convert to PIL Image
        img = Image.fromarray(image)
        # Save to appropriate directory
        save_path = data_dir / str(label) / f'train_{idx}.png'
        img.save(save_path)
        
        if idx % 1000 == 0:
            print(f"Processed {idx} training images...")
    
    print("\nProcessing test data...")
    # Save test images
    for idx, (image, label) in enumerate(zip(test_images, test_labels)):
        img = Image.fromarray(image)
        save_path = data_dir / str(label) / f'test_{idx}.png'
        img.save(save_path)
        
        if idx % 1000 == 0:
            print(f"Processed {idx} test images...")
    
    print("\nData preparation completed!")
    print(f"Images have been organized into directories at: {data_dir}")
    print("You can now run train_model.py")

if __name__ == "__main__":
    prepare_mnist_data() 