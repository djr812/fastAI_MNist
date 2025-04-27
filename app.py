from flask import Flask, request, render_template, jsonify
from fastai.vision.all import *
from PIL import Image
import io
import base64
import numpy as np
from train_model import MNISTResNet
from torchvision import transforms

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

@app.route('/', methods=['GET'])
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get the image data from the request
        image_data = request.form['image'].split(',')[1]
        image_bytes = base64.b64decode(image_data)
        
        # Convert to PIL Image and preprocess
        image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        # Resize to 28x28
        image = image.resize((28, 28))
        # Convert to tensor and normalize
        tensor_image = torch.tensor(np.array(image)).float().permute(2, 0, 1) / 255.0
        tensor_image = normalize_transform(tensor_image)
        
        # Make prediction
        with torch.no_grad():
            output = model(tensor_image.unsqueeze(0))
            pred = output.argmax().item()
        
        return jsonify({'prediction': str(pred)})
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True) 