from flask import Flask, request, render_template, jsonify
from fastai.vision.all import *
from PIL import Image
import io
import base64
import numpy as np
from train_model import MNISTResNet
from torchvision import transforms
import cv2

app = Flask(__name__)

application=app

# Define normalization transform
normalize_transform = transforms.Normalize(*imagenet_stats)

# Load the trained model
try:
    model = MNISTResNet()
    model.load_state_dict(torch.load('../mnist_data/mnist_model.pkl'))
    model.eval()
except Exception as e:
    print(f"Model not found or error loading model: {e}")
    print("Please train the model first using train_model.py")

def segment_digits(image):
    # Convert to grayscale
    gray = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)
    # Apply threshold
    _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)
    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Sort contours from left to right
    contours = sorted(contours, key=lambda c: cv2.boundingRect(c)[0])
    
    digit_images = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        # Add padding
        padding = 10
        x = max(0, x - padding)
        y = max(0, y - padding)
        w = min(gray.shape[1] - x, w + 2 * padding)
        h = min(gray.shape[0] - y, h + 2 * padding)
        
        # Extract digit
        digit = thresh[y:y+h, x:x+w]
        # Resize to 28x28
        digit = cv2.resize(digit, (28, 28))
        # Convert to RGB
        digit_rgb = cv2.cvtColor(digit, cv2.COLOR_GRAY2RGB)
        digit_images.append(digit_rgb)
    
    return digit_images

@app.route('/', methods=['GET'])
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get the image data from the request
        image_data = request.form['image'].split(',')[1]
        image_bytes = base64.b64decode(image_data)
        
        # Convert to PIL Image
        image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        
        # Segment the image into individual digits
        digit_images = segment_digits(image)
        
        if not digit_images:
            return jsonify({'error': 'No digits detected'})
        
        predictions = []
        for digit_img in digit_images:
            # Convert to tensor and normalize
            tensor_image = torch.tensor(digit_img).float().permute(2, 0, 1) / 255.0
            tensor_image = normalize_transform(tensor_image)
            
            # Make prediction
            with torch.no_grad():
                output = model(tensor_image.unsqueeze(0))
                pred = output.argmax().item()
                predictions.append(str(pred))
        
        return jsonify({'predictions': predictions})
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True) 