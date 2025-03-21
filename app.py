import os
from dotenv import load_dotenv
from flask import Flask, request, jsonify
import torch
import torchvision.transforms as T
from PIL import Image

# Load environment variables from .env
load_dotenv()

app = Flask(__name__)

SECRET_KEY = os.getenv("SECRET_KEY")
API_KEY = os.getenv("API_KEY")

# Suppose we have a segmentation model loaded here
# For demonstration, let's load a trivial model or a placeholder
# (Replace with your real segmentation model initialization)
model = None  # you would load your actual model here

@app.route('/secret', methods=['GET'])
def get_secret():
    """
    Endpoint to verify secret injection works.
    It simply returns the SECRET_KEY environment variable (for demonstration).
    In production, you wouldn't typically expose your secrets in an endpoint!
    """
    return jsonify({"secret_key": SECRET_KEY, "api_key": API_KEY})

@app.route('/predict', methods=['POST'])
def predict():
    """
    Example endpoint that receives an image and returns segmentation predictions.
    """
    if not model:
        return jsonify({"error": "Model not loaded"}), 500

    # Assume the request contains an image file
    if 'image' not in request.files:
        return jsonify({"error": "No image file provided"}), 400

    img_file = request.files['image']
    image = Image.open(img_file)

    # Preprocess image (transforms, etc.)
    transform = T.Compose([
        T.Resize((256, 256)),
        T.ToTensor()
    ])
    image_tensor = transform(image).unsqueeze(0)

    # Run inference
    with torch.no_grad():
        prediction = model(image_tensor)
    
    # Convert prediction to a simple JSON response, e.g., predicted mask
    # This is a placeholder
    predicted_mask = prediction.argmax(dim=1).cpu().numpy().tolist()

    return jsonify({"predicted_mask": predicted_mask})

if __name__ == '__main__':
    # Use 0.0.0.0 to run inside container
    app.run(host='0.0.0.0', port=5000)
