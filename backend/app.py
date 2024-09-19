from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image
import io
import os

app = Flask(__name__)
CORS(app)

print("----------------- Starting Backend Server -----------------")

# =========================
# Device Configuration
# =========================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# =========================
# Image Transformations
# =========================
IMG_HEIGHT = 100
IMG_WIDTH = 75

data_transforms = transforms.Compose([
    transforms.Resize((IMG_HEIGHT, IMG_WIDTH)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],   # Mean
                         [0.229, 0.224, 0.225])  # Std
])

# =========================
# Class Names Definitions
# =========================
class_names_skin = ['non-skin', 'skin']
class_names_malignancy = ['benign', 'malignant']
class_names_cancer_type = ['akiec', 'bcc', 'mel']

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
    model = models.resnet18(weights=None)
    num_ftrs = model.fc.in_features
    # Replace the final layer with a new one for the desired number of classes
    model.fc = nn.Linear(num_ftrs, num_classes)
    # Load the saved weights
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model

# =========================
# Load All Models
# =========================
print("Loading models...")

skin_model_path = '../weights/skin_nonskin_resnet.pth'
malignancy_model_path = '../weights/cancer_noncancer_model_10epochs.pth'
cancer_type_model_path = '../weights/malignant_model.pth'

# Check if model files exist
for path in [skin_model_path, malignancy_model_path, cancer_type_model_path]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model file not found: {path}")

skin_model = load_model(skin_model_path, num_classes=2)
malignancy_model = load_model(malignancy_model_path, num_classes=2)
cancer_type_model = load_model(cancer_type_model_path, num_classes=3)

print("All models loaded successfully.")

# =========================
# Preprocess Image Function
# =========================
def preprocess_image(image_bytes):
    img = Image.open(io.BytesIO(image_bytes))
    img = img.convert("RGB")
    return img

# =========================
# Hierarchical Inference Function
# =========================
def hierarchical_inference(image, skin_model, malignancy_model, cancer_type_model):
    """
    Performs hierarchical inference on a PIL image.
    
    Args:
        image (PIL.Image): Input image.
        skin_model, malignancy_model, cancer_type_model: Loaded models.
    
    Returns:
        result (dict): Dictionary containing predictions.
    """
    result = {}
    print("------------------------------------------")
    print("Stage 1: Skin Detection")
    stage = 'skin'
    input_skin = data_transforms(image).unsqueeze(0).to(device)
    with torch.no_grad():
        output_skin = skin_model(input_skin)
        probs_skin = torch.softmax(output_skin, dim=1)[0]
        _, preds_skin = torch.max(output_skin, 1)
        skin_prob = probs_skin[preds_skin].item()
        skin_prediction = class_names_skin[preds_skin.item()]
        # Compute entropy
        entropy_skin = -torch.sum(probs_skin * torch.log(probs_skin + 1e-8)).item()
        # Class probabilities
        class_probabilities_skin = {class_names_skin[i]: probs_skin[i].item() for i in range(len(class_names_skin))}
        # Collect result
        result = {
            "class_index": preds_skin.item(),
            "disease_name": skin_prediction,
            "class_probabilities": class_probabilities_skin,
            "uncertainty": entropy_skin,
            "heatmap_url": None,  # No heatmap
            "stage": "skin"
        }
    print(f"Skin Prediction: {skin_prediction} ({skin_prob:.2f})")
    print("------------------------------------------")

    if skin_prediction == 'skin':
        print("Skin detected. Proceeding to next stages.")
        print("------------------------------------------")
        print("Stage 2: Malignancy Classification")
        stage = 'malignancy'
        input_malignancy = data_transforms(image).unsqueeze(0).to(device)
        with torch.no_grad():
            output_malignancy = malignancy_model(input_malignancy)
            probs_malignancy = torch.softmax(output_malignancy, dim=1)[0]
            _, preds_malignancy = torch.max(output_malignancy, 1)
            malignancy_prob = probs_malignancy[preds_malignancy].item()
            malignancy_prediction = class_names_malignancy[preds_malignancy.item()]
            # Compute entropy
            entropy_malignancy = -torch.sum(probs_malignancy * torch.log(probs_malignancy + 1e-8)).item()
            # Class probabilities
            class_probabilities_malignancy = {class_names_malignancy[i]: probs_malignancy[i].item() for i in range(len(class_names_malignancy))}
            # Collect result
            result = {
                "class_index": preds_malignancy.item(),
                "disease_name": malignancy_prediction,
                "class_probabilities": class_probabilities_malignancy,
                "uncertainty": entropy_malignancy,
                "heatmap_url": None,  # No heatmap
                "stage": "malignancy"
            }
        print(f"Malignancy Prediction: {malignancy_prediction} ({malignancy_prob:.2f})")
        print("------------------------------------------")

        if malignancy_prediction == 'malignant':
            print("Malignant lesion detected. Proceeding to next stage.")
            print("------------------------------------------")
            print("Stage 3: Cancer Type Classification")
            stage = 'cancer_type'
            input_cancer_type = data_transforms(image).unsqueeze(0).to(device)
            with torch.no_grad():
                output_cancer_type = cancer_type_model(input_cancer_type)
                probs_cancer_type = torch.softmax(output_cancer_type, dim=1)[0]
                _, preds_cancer_type = torch.max(output_cancer_type, 1)
                cancer_type_prob = probs_cancer_type[preds_cancer_type].item()
                cancer_type_prediction = class_names_cancer_type[preds_cancer_type.item()]
                # Compute entropy
                entropy_cancer_type = -torch.sum(probs_cancer_type * torch.log(probs_cancer_type + 1e-8)).item()
                # Class probabilities
                class_probabilities_cancer_type = {class_names_cancer_type[i]: probs_cancer_type[i].item() for i in range(len(class_names_cancer_type))}
                # Collect result
                result = {
                    "class_index": preds_cancer_type.item(),
                    "disease_name": cancer_type_prediction,
                    "class_probabilities": class_probabilities_cancer_type,
                    "uncertainty": entropy_cancer_type,
                    "heatmap_url": None,  # No heatmap
                    "stage": "cancer_type"
                }
            print(f"Cancer Type Prediction: {cancer_type_prediction} ({cancer_type_prob:.2f})")
            print("------------------------------------------")
    
    # After all stages, set stage to 'model_predicted'
    if skin_prediction == 'skin' and (malignancy_prediction == 'benign' or (malignancy_prediction == 'malignant' and cancer_type_prediction in class_names_cancer_type)):
        result["stage"] = "model_predicted"
    
    return result

# =========================
# Predict Endpoint
# =========================
@app.route('/predict', methods=['POST']) 
def predict(): 
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    
    file = request.files['file']
    image_bytes = file.read()
    image = preprocess_image(image_bytes)
    
    try:
        predictions = hierarchical_inference(image, skin_model, malignancy_model, cancer_type_model)
        # Return predictions as JSON with required keys
        response = {
            "class_index": predictions.get("class_index"),
            "disease_name": predictions.get("disease_name"),
            "class_probabilities": predictions.get("class_probabilities"),
            "uncertainty": predictions.get("uncertainty"),
            "heatmap_url": None, # As per your request, no heatmap is generated
            "stage": predictions.get("stage")
        }
        return jsonify(response)

    except Exception as e:
        print(f"An error occurred during inference: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)