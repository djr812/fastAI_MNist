# MNIST Digit Recognition Web App

A web application that uses FastAI and PyTorch to recognize handwritten digits from the MNIST dataset. The application features a modern web interface where users can draw digits and get real-time predictions.

## Features

- Modern web interface for drawing digits
- Real-time digit prediction
- Support for multiple digit recognition
- Custom ResNet architecture with residual connections
- Advanced data augmentation
- Two-phase training process
- Automatic learning rate finding
- Model checkpointing
- Mobile-friendly interface with touch support

## Setup

1. Clone this repository
2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Download the MNIST dataset from [Kaggle](https://www.kaggle.com/datasets/hojjatk/mnist-dataset) and place the files in the `mnist_data` directory:
   - `train.csv`
   - `test.csv`

4. Train the model:
```bash
python train_model.py
```

5. Run the Flask application:
```bash
python app.py
```

6. Access the web interface at `http://localhost:5000` or `http://<server_ip>:5000`

## Model Architecture

The model uses a custom ResNet architecture with the following features:
- Residual connections for better gradient flow
- Batch normalization for stable training
- Dropout layers for regularization
- Adaptive pooling for better feature aggregation
- Cross-entropy loss with label smoothing

## Training Process

The training process includes:
1. Data augmentation:
   - Random rotation (±15 degrees)
   - Random zoom (up to 10%)
   - Random lighting changes
   - Random warping

2. Two-phase training:
   - Initial training (20 epochs)
   - Fine-tuning (10 epochs)
   - Learning rate scheduling
   - Model checkpointing

3. Automatic learning rate finding
4. Validation monitoring
5. Best model saving

## Project Structure

```
mnist/                    # Main project directory
├── app.py               # Flask web application
├── train_model.py       # Model training script
├── requirements.txt     # Python dependencies
└── templates/           # Web interface templates
    └── index.html      # Main web interface

mnist_data/              # Dataset directory 
├── train.csv           # Training data
└── test.csv            # Test data
```

## Multi-Digit Recognition

The application now supports recognizing multiple digits in a single drawing:
- Draw multiple digits on the canvas
- Click 'Predict' to get predictions for all digits
- Each digit is automatically segmented and processed
- Predictions are displayed in order from left to right
- Individual digit boxes show each prediction

## Notes

- The model is trained on the MNIST dataset, which consists of 28x28 grayscale images
- The web interface uses HTML5 Canvas for drawing
- The application uses FastAI's data augmentation pipeline
- The model is saved in PyTorch format
- The web interface is mobile-friendly and supports touch input

## Web Interface

1. Draw one or more digits (0-9) on the canvas
2. Click 'Predict' to get the model's prediction
3. Click 'Clear' to start over
4. The prediction will show all recognized digits in order
5. Individual digit boxes display each prediction separately

## Implementation Details

- The model uses a custom ResNet architecture
- Training includes data augmentation and learning rate scheduling
- The web interface uses HTML5 Canvas for drawing
- Digit segmentation is handled using OpenCV
- The application is built with Flask and FastAI 