from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import torch
import numpy as np
from PIL import Image
import io
import torch.nn as nn
import torch.nn.functional as F
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import tempfile
import os

app = Flask(__name__)
CORS(app)

print("----------------- Starting Backend Server -----------------")
# Check if GPU is available, otherwise fall back to CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

class SimpleCNN(nn.Module):
    def __init__(self, num_classes=7):
        super(SimpleCNN, self).__init__()
        # First convolutional layer
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.dropout1 = nn.Dropout(0.5)  # Apply dropout after first conv layer
        # Second convolutional layer
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.dropout2 = nn.Dropout(0.5)  # Apply dropout after second conv layer
        # Third convolutional layer
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.dropout3 = nn.Dropout(0.5)  # Apply dropout after third conv layer
        self.pool = nn.MaxPool2d(2, 2)  # 2x2 Max Pooling
        # Fully connected layers
        self.fc1 = nn.Linear(128 * 9 * 12, 512)  # Flatten after pooling
        self.dropout_fc1 = nn.Dropout(0.5)  # Apply dropout after first fully connected layer
        self.fc2 = nn.Linear(512, num_classes)  # Output layer
    def forward(self, x):
        # First conv layer + ReLU + Dropout + Pooling
        x = self.pool(F.relu(self.conv1(x)))
        x = self.dropout1(x)
        # Second conv layer + ReLU + Dropout + Pooling
        x = self.pool(F.relu(self.conv2(x)))
        x = self.dropout2(x)
        # Third conv layer + ReLU + Dropout + Pooling
        x = self.pool(F.relu(self.conv3(x)))
        x = self.dropout3(x)
        # Flatten the tensor for fully connected layers
        x = x.view(-1, 128 * 9 * 12)
        # Fully connected layer + ReLU + Dropout
        x = F.relu(self.fc1(x))
        x = self.dropout_fc1(x)
        # Output layer
        x = self.fc2(x)
        return x

# Initialize and load model
model = SimpleCNN(num_classes=7).to(device) 
# Load the trained model weights
model_file_path = '../weights/model_v1_acc_73.pth'
#model_file_path = '../weights/model_v3.pth'
model.load_state_dict(torch.load(model_file_path, weights_only=True))
model.eval() 
print(f"Model loaded successfully from {model_file_path}")

class SaveFeatures():
    features = None
    def __init__(self, m): self.hook = m.register_forward_hook(self.hook_fn)
    def hook_fn(self, module, input, output): self.features = ((output.cpu()).data).numpy()
    def remove(self): self.hook.remove()

# Function to preprocess image
def preprocess_image(image):
    img = Image.open(io.BytesIO(image))
    img = img.convert("RGB")
    img = img.resize((100, 75))
    img = np.array(img) / 255.0
    img = np.transpose(img, (2, 0, 1))
    img = np.expand_dims(img, axis=0)
    img = torch.tensor(img).float().to(device)    
    return img

# Function to perform Monte Carlo Dropout inference
def mc_dropout_inference(model, x, num_samples=100):
    model.train()  # Keep dropout layers active (necessary for Monte Carlo Dropout)
    outputs = []
    for _ in range(num_samples):
        with torch.no_grad():
            outputs.append(F.softmax(model(x), dim=1).unsqueeze(0))  # Collect predictions with softmax
    outputs = torch.cat(outputs, dim=0)  # Stack outputs for all samples
    mean_output = torch.mean(outputs, dim=0)  # Mean prediction (average over multiple runs)
    variance_output = torch.var(outputs, dim=0)  # Variance (uncertainty)

    return mean_output, variance_output

# Generate Class Activation Map (CAM) 
def getCAM(feature_conv, weight_fc, class_idx, target_size, threshold=0.9):
    _, nc, h, w = feature_conv.shape  # nc: number of channels, h: height, w: width
    cam = np.zeros((h, w), dtype=np.float32)
    # Multiply each channel by the corresponding weight, limit to available channels
    for i in range(min(nc, len(weight_fc[class_idx]))):  
        cam += weight_fc[class_idx][i] * feature_conv[0, i, :, :]
    # Normalize the CAM
    cam = cam - np.min(cam)
    cam_img = cam / np.max(cam)
    # Apply thresholding: Set values below the threshold to 0
    cam_img[cam_img < threshold] = 0
    # Resize the CAM to match the input image size (use bilinear interpolation)
    cam_img = cv2.resize(cam_img, target_size, interpolation=cv2.INTER_LINEAR)

    return cam_img
 
@app.route('/predict', methods=['POST']) 
def predict(): 
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    
    file = request.files['file']
    image = file.read()
    processed_image = preprocess_image(image)

    lesion_type_index = {
        0: 'Melanocytic nevi', 1: 'Melanoma', 2: 'Benign keratosis-like lesions',
        3: 'Basal cell carcinoma', 4: 'Actinic keratoses', 5: 'Vascular lesions',
        6: 'Dermatofibroma'
    }
    ################
    print("Performing Monte Carlo Dropout Inference")
    test_sample = processed_image[0].unsqueeze(0)  # Select the first test sample and add a batch dimension
    print(f"Test sample shape: {test_sample.shape}")
    print(f"Test sample type: {type(test_sample)}")
    print(f"Test sample dtype: {test_sample.dtype}")

    mean_output, variance_output = mc_dropout_inference(model, test_sample, num_samples=100)
    print(f"Raw Variance: {variance_output}") 
    # Get the predicted class index for the sample
    predicted_class_idx = torch.argmax(mean_output, dim=1).item()
    predicted_class_name = lesion_type_index[predicted_class_idx]
    # Create a dictionary to map each class to its mean probability
    class_probabilities = {lesion_type_index[i]: float(mean_output[0][i]) for i in range(len(mean_output[0]))}
    # Uncertainty (variance) for the predicted class
    predicted_class_uncertainty = variance_output[0][predicted_class_idx].item() * 100  # Scaling to percentage
    predicted_class_uncertainty = round(predicted_class_uncertainty, 2)
    print(f"Uncertainty for {predicted_class_name}: {predicted_class_uncertainty:.0f}%")
    ##################

    print("\n------------------ Model Output -----------------------")
    print(f"Model Weights: {model_file_path}")
    print(f"Mean Prediction: {mean_output}")
    print(f"Variance (Uncertainty): {variance_output}")
    print(f"Predicted Class: {predicted_class_name}")
    print(f"Class Probabilities: {class_probabilities}")
    print(f"Uncertainty for {predicted_class_name}: {predicted_class_uncertainty:.0f}%")
    print("-----------------------------------------\n")

    processed_image = test_sample
    # Generate heatmap (CAM)
    final_layer = model.conv3
    activated_features = SaveFeatures(final_layer)
    prediction = model(processed_image)
    pred_probabilities = F.softmax(prediction, dim=1).data.squeeze()
    activated_features.remove()
    # Get the top class index
    _, top_class_idx = torch.topk(pred_probabilities, 1)
    # Resize CAM to match input image size (e.g., 75x100 from your input shape)
    input_image_size = (100, 75)  # (width, height)
    weight_softmax_params = list(model.fc2.parameters())
    weight_softmax = np.squeeze(weight_softmax_params[0].cpu().data.numpy())
    # Get the top class index
    class_idx = top_class_idx.item()
    cam_overlay = getCAM(activated_features.features, weight_softmax, class_idx, input_image_size, threshold=0.6)
    # Prepare image for heatmap overlay
    original_image = np.transpose(processed_image.cpu().numpy()[0], (1, 2, 0))
    original_image = original_image - np.min(original_image)
    original_image = original_image / np.max(original_image)

    # Plot the normal image and the heatmap overlay side by side
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    # Show the original image
    ax[0].imshow(original_image)
    ax[0].axis('off')
    ax[0].set_title("Original Image")
    # Adjust transparency and colormap for better visualization 
    alpha_value = 0.6
    # Lower transparency for a better overlay
    ax[1].imshow(original_image)
    ax[1].imshow(cam_overlay, alpha=alpha_value, cmap='viridis')  # Try different colormaps like 'plasma' or 'viridis'
    ax[1].axis('off')
    ax[1].set_title("Image with Heatmap (Thresholded)")

    # Save heatmap to temporary file
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.png')
    plt.savefig(temp_file.name, bbox_inches='tight', pad_inches=0)
    temp_file.seek(0)
    # Save heatmap in a permanent location
    heatmap_filename = f"heatmap_{os.path.basename(temp_file.name)}"
    heatmap_path = os.path.join('static', 'heatmaps', heatmap_filename)
    plt.savefig(heatmap_path, bbox_inches='tight', pad_inches=0)
    plt.close()  # Close the figure after saving

    # Return the heatmap URL relative to the static folder
    return jsonify({
        "class_index": int(predicted_class_idx),
        "disease_name": predicted_class_name,
        "class_probabilities": class_probabilities,
        "uncertainty": float(predicted_class_uncertainty),
        "heatmap_url": f"/static/heatmaps/{heatmap_filename}"
    })

@app.route('/heatmap/<filename>', methods=['GET'])
def serve_heatmap(filename):
    # Serve the heatmap from the static/heatmaps directory
    return send_file(os.path.join('static', 'heatmaps', filename), mimetype='image/png')

if __name__ == '__main__':
    app.run(debug=True)
