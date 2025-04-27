# MNIST Digit Recognition Web App

This project implements a web-based handwritten digit recognition system using FastAI and Flask. Users can draw digits on a canvas, and the trained model will predict the digit.

## Setup

1. Install the required dependencies:
```bash
pip install -r requirements.txt
```

2. Prepare the MNIST dataset:
   a. Download the MNIST dataset files from http://yann.lecun.com/exdb/mnist/
   b. Place the following files in the `../mnist_data` directory:
      - train-images-idx3-ubyte.gz
      - train-labels-idx1-ubyte.gz
   c. Run the data preparation script:
   ```bash
   python prepare_data.py
   ```
   This will organize the images into the required directory structure.

3. Train the model:
```bash
python train_model.py
```

4. Run the web application:
```bash
python app.py
```

5. Open your web browser and navigate to `http://localhost:5000`

## Usage

1. Draw a digit (0-9) on the canvas using your mouse
2. Click "Predict" to get the model's prediction
3. Use "Clear" to erase the canvas and try another digit

## Project Structure

- `prepare_data.py`: Script to prepare MNIST data into the required format
- `train_model.py`: Script for training the CNN model on MNIST data
- `app.py`: Flask web application
- `templates/index.html`: Web interface
- `mnist_model.pkl`: Trained model (generated after training)

## Requirements

See `requirements.txt` for the complete list of dependencies. 