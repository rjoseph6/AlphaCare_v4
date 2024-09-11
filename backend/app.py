from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
import numpy as np
from PIL import Image
import io
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

app = Flask(__name__)
CORS(app)  # Enable Cross-Origin Resource Sharing

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

# Initialize the model architecture
model = SimpleCNN(num_classes=7).to(device)
# Load the state_dict into the model
model.load_state_dict(torch.load('model_v3.pth', weights_only=True))
# Set the model to evaluation mode
#model.eval()

print("PyTorch Model loaded successfully!")

# Monte Carlo Dropout for uncertainty quantification (PyTorch)
def monte_carlo_dropout(model, x_input, n_iter=100):
    model.train()  # Enable dropout during inference
    predictions = np.array([model(x_input).detach().numpy() for _ in range(n_iter)])
    mean_prediction = np.mean(predictions, axis=0)
    variance_prediction = np.var(predictions, axis=0)
    model.eval()  # Switch back to evaluation mode
    return mean_prediction, variance_prediction

# Preprocess the image to be fed into the PyTorch model
def preprocess_image(image):
    img = Image.open(io.BytesIO(image))
    img = img.convert("RGB")  # Ensure RGB
    img = img.resize((100, 75))  # Resize to match model's expected input size (e.g., 100x75)
    img = np.array(img) / 255.0  # Normalize pixel values
    img = np.transpose(img, (2, 0, 1))  # PyTorch expects channel-first format
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    img = torch.tensor(img).float()  # Convert to tensor
    return img

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    
    file = request.files['file']
    image = file.read()
    processed_image = preprocess_image(image)

    # Define the dictionary for disease classes
    lesion_type_index = {
        0: 'Melanocytic nevi',
        1: 'Melanoma',
        2: 'Benign keratosis-like lesions',
        3: 'Basal cell carcinoma',
        4: 'Actinic keratoses',
        5: 'Vascular lesions',
        6: 'Dermatofibroma'
    }

    # Use Monte Carlo Dropout for uncertainty quantification
    mean_prediction, variance_prediction = monte_carlo_dropout(model, processed_image, n_iter=100)

    # Get the predicted class (index of the highest mean probability)
    predicted_class = np.argmax(mean_prediction[0])
    predicted_class_name = lesion_type_index[predicted_class]

    # Create a dictionary to map each class to its mean probability
    class_probabilities = {lesion_type_index[i]: float(mean_prediction[0][i]) for i in range(len(mean_prediction[0]))}

    # Uncertainty (variance) for the predicted class
    uncertainty = variance_prediction[0][predicted_class]
    uncertainty = round(uncertainty * 100, 2)  # Round the uncertainty to 3 decimal places

    print("\n------------------ Model Output -----------------------")
    # Print debugging info for backend console
    print(f"Mean Prediction: {mean_prediction}")
    print(f"Variance (Uncertainty): {variance_prediction}")
    print(f"Predicted Class: {predicted_class}")
    print(f"Prediction: {predicted_class_name}")
    print(f"Class Probabilities: {class_probabilities}")
    print(f"Uncertainty for {predicted_class_name}: {uncertainty}%")
    print("-----------------------------------------\n")

    # Return the predicted class, probabilities, and uncertainty for each class
    return jsonify({
        "class_index": int(predicted_class),
        "disease_name": predicted_class_name,
        "class_probabilities": class_probabilities,
        "uncertainty": float(uncertainty)  # Uncertainty for the predicted class
    })

if __name__ == '__main__':
    app.run(debug=True)




'''from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
from flask_cors import CORS
import numpy as np
from PIL import Image
import io
import tensorflow as tf
from tensorflow.keras.optimizers.legacy import SGD
import keras

app = Flask(__name__)
CORS(app)  # Enable Cross-Origin Resource Sharing

# Load pretrained CNN model (adjust the path as needed)
#model = load_model('model_4.keras')
new_model = tf.keras.models.load_model('model_h5.h5', compile=False)

#model = load_model('model_h5.h5')
print("Model loaded successfully!")

# multiple forward passes with Dropout for Monte Carlo method
def monte_carlo_dropout(model, x_input, n_iter=100):
    # store predictions from multiple forward passes
    predictions = np.array([model(x_input, training=True) for _ in range(n_iter)])
    # mean of predictions
    mean_prediction = np.mean(predictions, axis=0)
    # variance of the predictions (uncertainty)
    variance_prediction = np.var(predictions, axis=0)
    return mean_prediction, variance_prediction

def preprocess_image(image):
    img = Image.open(io.BytesIO(image))
    img = img.convert("RGB")  # Ensure RGB
    img = img.resize((100, 75))  # Resize to model's expected input size (e.g., 100x75)
    img = np.array(img) / 255.0  # Normalize pixel values
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    return img

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    
    file = request.files['file']
    image = file.read()
    processed_image = preprocess_image(image)

    # Define the dictionary for disease classes
    lesion_type_index = {
        0: 'Melanocytic nevi',
        1: 'Melanoma',
        2: 'Benign keratosis-like lesions',
        3: 'Basal cell carcinoma',
        4: 'Actinic keratoses',
        5: 'Vascular lesions',
        6: 'Dermatofibroma'
    }

    # Use Monte Carlo Dropout for uncertainty quantification
    mean_prediction, variance_prediction = monte_carlo_dropout(model, processed_image, n_iter=100)

    # Get the predicted class (index of the highest mean probability)
    predicted_class = np.argmax(mean_prediction[0])
    predicted_class_name = lesion_type_index[predicted_class]

    # Create a dictionary to map each class to its mean probability
    class_probabilities = {lesion_type_index[i]: float(mean_prediction[0][i]) for i in range(len(mean_prediction[0]))}

    # Uncertainty (variance) for the predicted class
    uncertainty = variance_prediction[0][predicted_class]
    # Round the uncertainty to 3 decimal places
    uncertainty = round(uncertainty * 100, 2)

    print("\n------------------ Model Output -----------------------")
    # Print debugging info for backend console
    print(f"Mean Prediction: {mean_prediction}")
    print(f"Variance (Uncertainty): {variance_prediction}")
    print(f"Predicted Class: {predicted_class}")
    print(f"Prediction: {predicted_class_name}")
    print(f"Class Probabilities: {class_probabilities}")
    print(f"Uncertainty for {predicted_class_name}: {uncertainty}%")
    print("-----------------------------------------\n")

    # Return the predicted class, probabilities, and uncertainty for each class
    return jsonify({
        "class_index": int(predicted_class),
        "disease_name": predicted_class_name,
        "class_probabilities": class_probabilities,
        "uncertainty": float(uncertainty)  # Uncertainty for the predicted class
    })

if __name__ == '__main__':
    app.run(debug=True)'''