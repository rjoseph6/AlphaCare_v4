from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
from flask_cors import CORS
import numpy as np
from PIL import Image
import io

app = Flask(__name__)
CORS(app)  # Enable Cross-Origin Resource Sharing

# Load pretrained CNN model (adjust the path as needed)
model = load_model('model_4.keras')
#model = load_model('model_new_3.keras', compile=True)
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

    print("\n-----------------------------------------")
    # Print debugging info for backend console
    #print(f"Mean Prediction: {mean_prediction}")
    #print(f"Variance (Uncertainty): {variance_prediction}")
    #print(f"Predicted Class: {predicted_class}")
    print(f"Prediction: {predicted_class_name}")
    #print(f"Class Probabilities: {class_probabilities}")
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