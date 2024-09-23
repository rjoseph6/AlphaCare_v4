# src/app.py

from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image
import io
import os
import numpy as np
from scipy.stats import entropy  # Import entropy for uncertainty calculation

app = Flask(__name__)
CORS(app)

print("----------------- Starting Backend Server -----------------")

# =========================
# Device Configuration
# =========================
if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("Using MPS device")
elif torch.cuda.is_available():
    device = torch.device("cuda")
    print("Using CUDA device")
else:
    device = torch.device("cpu")
    print("Using CPU device")

# =========================
# Image Transformations
# =========================
IMG_HEIGHT = 224  # Updated to 224 for ResNet
IMG_WIDTH = 224   # Updated to 224 for ResNet

data_transforms = transforms.Compose([
    transforms.Resize((IMG_HEIGHT, IMG_WIDTH)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],   # Mean for normalization
                         [0.229, 0.224, 0.225])  # Std for normalization
])

# =========================
# Class Names Definitions
# =========================
lesion_type_dict = {
    0: 'Actinic Keratosis',
    1: 'Basal Cell Carcinoma',
    2: 'Melanoma',
    3: 'Benign Keratosis-like Lesions',
    4: 'Dermatofibroma',
    5: 'Melanocytic Nevi',
    6: 'Vascular Lesions'
}

# =========================
# Model Loading Function
# =========================
def load_model(model_path, num_classes):
    """
    Loads a ResNet18 model, modifies the final layer, and loads the saved weights.

    Args:
        model_path (str): Path to the saved model weights.
        num_classes (int): Number of output classes.

    Returns:
        model (torch.nn.Module): The loaded and modified model.
    """
    # Load pre-trained ResNet18 model
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    num_ftrs = model.fc.in_features
    # Replace the final layer with a new one for the desired number of classes
    model.fc = nn.Linear(num_ftrs, num_classes)
    # Load the saved weights
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model

# =========================
# Load the Trained Model
# =========================
print("Loading the 7-class ResNet-18 model...")

model_path = '../weights/resnet18_ham10000_7classes.pth'

# Check if the model file exists
if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model file not found: {model_path}")

# Load the model
model = load_model(model_path, num_classes=7)

print("Model loaded successfully.")

# =========================
# Preprocess Image Function
# =========================
def preprocess_image(image_bytes):
    """
    Preprocesses the input image bytes for model inference.

    Args:
        image_bytes (bytes): Image in bytes.

    Returns:
        img (PIL.Image): Preprocessed PIL Image.
    """
    try:
        img = Image.open(io.BytesIO(image_bytes))
        img = img.convert("RGB")
        return img
    except Exception as e:
        print(f"Error in image preprocessing: {e}")
        return None

# =========================
# Predict Endpoint
# =========================
@app.route('/predict', methods=['POST'])
def predict():
    """
    Endpoint to handle image classification requests.

    Expects:
        - 'file': Image file uploaded via form-data.

    Returns:
        JSON response with class indices, full disease names, probabilities, and uncertainty.
    """
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files['file']

    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    image_bytes = file.read()
    image = preprocess_image(image_bytes)

    if image is None:
        return jsonify({"error": "Invalid image file"}), 400

    try:
        # Preprocess the image
        input_tensor = data_transforms(image).unsqueeze(0).to(device)

        # Forward pass
        with torch.no_grad():
            outputs = model(input_tensor)
            probabilities = torch.softmax(outputs, dim=1).cpu().numpy()[0]

        # Map class indices to full names and probabilities
        predictions = {}
        for idx, prob in enumerate(probabilities):
            disease_full_name = lesion_type_dict.get(idx, f"Class {idx}")
            predictions[disease_full_name] = round(float(prob), 4)  # Rounded to 4 decimal places

        # Identify the predicted class
        predicted_class_idx = np.argmax(probabilities)
        predicted_class_name = lesion_type_dict.get(predicted_class_idx, f"Class {predicted_class_idx}")
        predicted_probability = round(float(probabilities[predicted_class_idx]), 4)

        # Compute Uncertainty using Entropy
        uncertainty = round(float(entropy(probabilities, base=2)), 4)  # Entropy in bits

        print("\n----------------- Prediction Results -----------------")
        print(f"Predicted Class Index: {predicted_class_idx}")
        print(f"Predicted Disease Name: {predicted_class_name}")
        print(f"Predicted Probability: {predicted_probability}%")
        print(f"Uncertainty: {uncertainty}")
        print(f"Class Probabilities: {predictions}")
        print("------------------------------------------------------\n")
        # Prepare the response
        response = {
            "predicted_class_index": int(predicted_class_idx),
            "predicted_disease_name": predicted_class_name,
            "predicted_probability": predicted_probability,
            "class_probabilities": predictions,
            "uncertainty": uncertainty,  # Added uncertainty field
            "heatmap_url": None  # Placeholder if you implement heatmap generation
        }

        print(f"Prediction: {response}")

        return jsonify(response), 200

    except Exception as e:
        print(f"An error occurred during inference: {e}")
        return jsonify({"error": str(e)}), 500

# =========================
# Run the Flask App
# =========================
if __name__ == '__main__':
    # Set debug=False in production
    app.run(host='0.0.0.0', port=5000, debug=True)