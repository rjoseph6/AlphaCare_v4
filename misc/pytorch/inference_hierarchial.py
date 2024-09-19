import os
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import matplotlib.pyplot as plt

# =========================
# Device Configuration
# =========================
if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("Using MPS device")
else:
    device = torch.device("cpu")
    print("Using CPU device")

# =========================
# Image Transformations
# =========================
IMG_HEIGHT = 100
IMG_WIDTH = 75

# Define the same transformations used during training for each classifier
data_transforms = {
    'skin': transforms.Compose([
        transforms.Resize((IMG_HEIGHT, IMG_WIDTH)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],   # Mean
                             [0.229, 0.224, 0.225])  # Std
    ]),
    'malignancy': transforms.Compose([
        transforms.Resize((IMG_HEIGHT, IMG_WIDTH)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ]),
    'cancer_type': transforms.Compose([
        transforms.Resize((IMG_HEIGHT, IMG_WIDTH)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ]),
}

# =========================
# Class Names Definitions
# =========================
class_names_skin = ['non-skin', 'skin']  # Adjust based on training
class_names_malignancy = ['benign', 'malignant']  # Adjust based on training
class_names_cancer_type = ['akiec', 'bcc', 'mel']  # Adjust based on training

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
    # Replace the final layer with a new one (for the desired number of classes)
    model.fc = nn.Linear(num_ftrs, num_classes)
    # Load the saved weights
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model

# =========================
# Load All Models
# =========================
def load_all_models():
    """
    Loads all three models required for the hierarchical inference.
    
    Returns:
        skin_model, malignancy_model, cancer_type_model
    """
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
    return skin_model, malignancy_model, cancer_type_model

# =========================
# Inference Function
# =========================
def hierarchical_inference(image_path, skin_model, malignancy_model, cancer_type_model):
    """
    Performs hierarchical inference on a single image.
    
    Args:
        image_path (str): Path to the input image.
        skin_model, malignancy_model, cancer_type_model: Loaded models.
    
    Returns:
        result (dict): Dictionary containing predictions at each stage.
    """
    # Check if image exists
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Test image not found at {image_path}")
    
    # Load image
    image = Image.open(image_path).convert('RGB')
    print("Loaded image successfully.")
    # =========================
    # Stage 1: Skin Detection
    # =========================
    print("------------------------------------------")
    print("Stage 1: Skin Detection")
    input_skin = data_transforms['skin'](image).unsqueeze(0).to(device)  # Add batch dimension
    with torch.no_grad():
        output_skin = skin_model(input_skin)
        probs_skin = torch.softmax(output_skin, dim=1)
        _, preds_skin = torch.max(output_skin, 1)
        skin_prob = probs_skin[0][preds_skin].item()
        skin_prediction = class_names_skin[preds_skin.item()]
    
    result = {'skin': {'prediction': skin_prediction, 'confidence': skin_prob}}
    print(f"Skin Prediction: {skin_prediction} ({skin_prob:.2f})")
    print("------------------------------------------")

    if skin_prediction == 'skin':
        print("Skin detected. Proceeding to next stages.")
        # =========================
        # Stage 2: Malignancy Classification
        # =========================
        print("------------------------------------------")
        print("Stage 2: Malignancy Classification")
        input_malignancy = data_transforms['malignancy'](image).unsqueeze(0).to(device)
        with torch.no_grad():
            output_malignancy = malignancy_model(input_malignancy)
            probs_malignancy = torch.softmax(output_malignancy, dim=1)
            _, preds_malignancy = torch.max(output_malignancy, 1)
            malignancy_prob = probs_malignancy[0][preds_malignancy].item()
            malignancy_prediction = class_names_malignancy[preds_malignancy.item()]
        
        result['malignancy'] = {'prediction': malignancy_prediction, 'confidence': malignancy_prob}
        print(f"Malignancy Prediction: {malignancy_prediction} ({malignancy_prob:.2f})")
        print("------------------------------------------")

        if malignancy_prediction == 'malignant':
            print("Malignant lesion detected. Proceeding to next stage.")
            # =========================
            # Stage 3: Cancer Type Classification
            # =========================
            print("------------------------------------------")
            print("Stage 3: Cancer Type Classification")
            input_cancer_type = data_transforms['cancer_type'](image).unsqueeze(0).to(device)
            with torch.no_grad():
                output_cancer_type = cancer_type_model(input_cancer_type)
                probs_cancer_type = torch.softmax(output_cancer_type, dim=1)
                _, preds_cancer_type = torch.max(output_cancer_type, 1)
                cancer_type_prob = probs_cancer_type[0][preds_cancer_type].item()
                cancer_type_prediction = class_names_cancer_type[preds_cancer_type.item()]
            
            result['cancer_type'] = {'prediction': cancer_type_prediction, 'confidence': cancer_type_prob}
            print(f"Cancer Type Prediction: {cancer_type_prediction} ({cancer_type_prob:.2f})")
            print("------------------------------------------")
    
    return result

# =========================
# Visualization Function (Optional)
# =========================
def visualize_predictions(image_path, predictions):
    """
    Displays the image with predictions annotated.
    
    Args:
        image_path (str): Path to the input image.
        predictions (dict): Predictions from each inference stage.
    """
    image = Image.open(image_path).convert('RGB')
    plt.figure(figsize=(6, 6))
    plt.imshow(image)
    plt.axis('off')
    
    # Prepare annotation text
    text = ""
    if 'skin' in predictions:
        skin_pred = predictions['skin']['prediction']
        skin_conf = predictions['skin']['confidence']
        text += f"Skin: {skin_pred} ({skin_conf:.2f})\n"
    if 'malignancy' in predictions:
        malignancy_pred = predictions['malignancy']['prediction']
        malignancy_conf = predictions['malignancy']['confidence']
        text += f"Malignancy: {malignancy_pred} ({malignancy_conf:.2f})\n"
    if 'cancer_type' in predictions:
        cancer_type_pred = predictions['cancer_type']['prediction']
        cancer_type_conf = predictions['cancer_type']['confidence']
        text += f"Cancer Type: {cancer_type_pred} ({cancer_type_conf:.2f})"
    
    # Add text to the image
    plt.text(10, IMG_HEIGHT - 10, text, fontsize=12, color='yellow',
             bbox=dict(facecolor='black', alpha=0.5))
    
    plt.show()

# =========================
# Main Function
# =========================
def main():
    # Load all models
    skin_model, malignancy_model, cancer_type_model = load_all_models()
    
    # Path to the test image
    #test_image_path = 'shading.jpeg'  # Replace with your test image path
    test_image_path = 'ISIC_0024317.jpg'  # Replace with your test image path

    # Perform hierarchical inference
    try:
        predictions = hierarchical_inference(test_image_path, skin_model, malignancy_model, cancer_type_model)
        print("\nInference Results:")
        for stage, pred_info in predictions.items():
            print(f"{stage.capitalize()}: {pred_info['prediction']} (Confidence: {pred_info['confidence']:.4f})")
        
        # Optional: Visualize the predictions
        visualize_predictions(test_image_path, predictions)
    
    except FileNotFoundError as e:
        print(e)
    except Exception as e:
        print(f"An error occurred during inference: {e}")

if __name__ == '__main__':
    main()