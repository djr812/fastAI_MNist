# MNIST Digit Recognition Web App

This project implements a web-based handwritten digit recognition system using FastAI and Flask. Users can draw digits on a canvas, and the trained model will predict the digit. The system uses a custom ResNet architecture optimized for MNIST digit recognition.

## Features

- Modern web interface with drawing canvas
- Real-time digit prediction
- Custom ResNet architecture with residual connections
- Advanced data augmentation for better generalization
- Two-phase training process (main training + fine-tuning)
- Automatic learning rate finding and scheduling
- Model checkpointing to save the best performing model

## Setup

1. Install the required dependencies:
```bash
pip install -r requirements.txt
```

2. Prepare the MNIST dataset:
   a. Download the MNIST dataset files from https://www.kaggle.com/datasets/hojjatk/mnist-dataset
   b. Place the following files in the `../mnist_data` directory:
      - train-images.idx3-ubyte
      - train-labels.idx1-ubyte
      - t10k-images.idx3-ubyte
      - t10k-labels.idx1-ubyte
   c. Run the data preparation script:
   ```bash
   python prepare_data.py
   ```
   This will organize the images into the required directory structure.

3. Train the model:
```bash
python train_model.py
```
   The training process includes:
   - Automatic learning rate finding
   - 20 epochs of initial training
   - 10 epochs of fine-tuning
   - Model checkpointing to save the best model
   - Progress tracking with training and validation metrics

4. Run the web application:
```bash
python app.py
```

5. Open your web browser and navigate to `http://localhost:5000` (or your server's IP address)

## Usage

1. Draw a digit (0-9) on the canvas using your mouse
2. Click "Predict" to get the model's prediction
3. Use "Clear" to erase the canvas and try another digit

## Model Architecture

The system uses a custom ResNet architecture specifically designed for MNIST digit recognition:

- Initial convolution block with batch normalization and dropout
- Three residual blocks with increasing channel depth
- Global average pooling
- Final fully connected layer with dropout
- Cross-entropy loss with label smoothing

## Training Process

1. Data Augmentation:
   - Rotation (±15 degrees)
   - Zoom (up to 10%)
   - Lighting adjustments
   - Warping
   - Affine transformations

2. Training Phases:
   - Learning rate finding
   - Main training (20 epochs)
   - Fine-tuning (10 epochs)
   - Automatic learning rate reduction on plateau
   - Best model checkpointing

## Project Structure

- `prepare_data.py`: Script to prepare MNIST data into the required format
- `train_model.py`: Script for training the CNN model with custom ResNet architecture
- `app.py`: Flask web application with real-time prediction
- `templates/index.html`: Web interface with drawing canvas
- `mnist_model.pkl`: Trained model weights (generated after training)

## Requirements

See `requirements.txt` for the complete list of dependencies.

## Notes

- The model is trained on the MNIST dataset and optimized for handwritten digit recognition
- The web interface provides a 280x280 pixel canvas for drawing
- Input images are automatically resized and normalized before prediction
- The model uses a batch size of 64 for stable training
- Training progress and metrics are displayed during the training process 