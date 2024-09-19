import React, { useState } from 'react';
import axios from 'axios';
import './App.css'; // Import custom CSS for styling
import { FaFileAlt, FaCheckCircle, FaMicroscope, FaBrain, FaChartBar, FaExclamationCircle } from 'react-icons/fa'; // Import icons

function App() {
  const [file, setFile] = useState(null);
  const [imagePreviewUrl, setImagePreviewUrl] = useState('');
  const [result, setResult] = useState('');
  const [probabilities, setProbabilities] = useState([]);
  const [uncertainty, setUncertainty] = useState(null);
  const [currentStage, setCurrentStage] = useState('');
  const [isLoading, setIsLoading] = useState(false);

  // Handle file upload and preview
  const handleFileChange = (event) => {
    const file = event.target.files[0];
    setFile(file);

    // Create image preview URL
    const reader = new FileReader();
    reader.onloadend = () => {
      setImagePreviewUrl(reader.result);
    };
    reader.readAsDataURL(file);

    // Reset states when a new file is selected
    setResult('');
    setProbabilities([]);
    setUncertainty(null);
    setCurrentStage('image_uploaded');
  };

  // Handle form submission to get predictions
  const handleSubmit = async () => {
    if (!file) {
      alert('Please upload an image first.');
      return;
    }

    const formData = new FormData();
    formData.append('file', file);

    setIsLoading(true);
    setCurrentStage('skin'); // Start with Skin Detection

    try {
      // Send request to Flask backend
      const response = await axios.post('http://localhost:5000/predict', formData, {
        headers: { 'Content-Type': 'multipart/form-data' },
      });

      const data = response.data;
      setResult(data.disease_name);
      setUncertainty(data.uncertainty);
      setProbabilities(Object.entries(data.class_probabilities));
      setCurrentStage(data.stage);
    } catch (error) {
      console.error('Error:', error);
      setResult('Error occurred while predicting');
      setCurrentStage('error');
    } finally {
      setIsLoading(false);
    }
  };

  // Function to determine the status of each step
  const getStepStatus = (step) => {
    const stages = ['image_uploaded', 'skin', 'malignancy', 'cancer_type', 'model_predicted'];
    const currentIndex = stages.indexOf(currentStage);
    const stepIndex = stages.indexOf(step);

    if (currentStage === 'error') {
      return 'error';
    } else if (stepIndex < currentIndex) {
      return 'completed';
    } else if (stepIndex === currentIndex) {
      return isLoading ? 'active' : 'completed';
    } else {
      return 'pending';
    }
  };

  // Function to get the icon for each step
  const getStepIcon = (step, status) => {
    switch (step) {
      case 'image_uploaded':
        return <FaFileAlt />;
      case 'skin':
        return <FaMicroscope />;
      case 'malignancy':
        return <FaBrain />;
      case 'cancer_type':
        return <FaChartBar />;
      case 'model_predicted':
        return <FaCheckCircle />;
      default:
        return null;
    }
  };

  // Function to get error icon
  const getErrorIcon = () => {
    return <FaExclamationCircle />;
  };

  return (
    <div className="app-container">
      <h1>Hierarchical Image Classifier</h1>

      {/* File upload and button */}
      <div className="upload-section">
        <input type="file" onChange={handleFileChange} />
        <button onClick={handleSubmit} disabled={!file || isLoading}>
          {isLoading ? 'Processing...' : 'Upload and Predict'}
        </button>
      </div>

      {/* Display Image and Stepper */}
      {imagePreviewUrl && (
        <div className="content-section">
          {/* Image Preview */}
          <div className="image-preview">
            <h3>Uploaded Image</h3>
            <img src={imagePreviewUrl} alt="Uploaded Preview" />
          </div>

          {/* Stepper Visualization */}
          <div className="stepper-visualization">
            <h3>Classification Steps</h3>
            <div className={`stepper ${currentStage === 'model_predicted' ? 'completed' : ''}`}>
              {/* Step 1: Image Uploaded */}
              <div className={`step ${getStepStatus('image_uploaded')}`}>
                <div className="step-icon">
                  {getStepIcon('image_uploaded', getStepStatus('image_uploaded'))}
                </div>
                <div className="step-title">Image Uploaded</div>
              </div>

              {/* Connector */}
              <div className={`connector ${getStepStatus('image_uploaded') === 'completed' ? 'completed' : ''}`}></div>

              {/* Step 2: Skin Detection */}
              <div className={`step ${getStepStatus('skin')}`}>
                <div className="step-icon">
                  {getStepIcon('skin', getStepStatus('skin'))}
                </div>
                <div className="step-title">Skin Detection</div>
                <div className="step-subtitle">Skin / Non-skin</div>
              </div>

              {/* Connector */}
              <div className={`connector ${getStepStatus('skin') === 'completed' ? 'completed' : ''}`}></div>

              {/* Step 3: Malignancy Classification */}
              <div className={`step ${getStepStatus('malignancy')}`}>
                <div className="step-icon">
                  {getStepIcon('malignancy', getStepStatus('malignancy'))}
                </div>
                <div className="step-title">Malignancy Classification</div>
                <div className="step-subtitle">Benign / Malignant</div>
              </div>

              {/* Connector */}
              <div className={`connector ${getStepStatus('malignancy') === 'completed' ? 'completed' : ''}`}></div>

              {/* Step 4: Cancer Type Classification */}
              <div className={`step ${getStepStatus('cancer_type')}`}>
                <div className="step-icon">
                  {getStepIcon('cancer_type', getStepStatus('cancer_type'))}
                </div>
                <div className="step-title">Cancer Type Classification</div>
                <div className="step-subtitle">akiec / bcc / mel</div>
              </div>

              {/* Connector */}
              <div className={`connector ${getStepStatus('cancer_type') === 'completed' ? 'completed' : ''}`}></div>

              {/* Step 5: Model Predicted */}
              <div className={`step ${getStepStatus('model_predicted')}`}>
                <div className="step-icon">
                  {currentStage === 'error' ? getErrorIcon() : getStepIcon('model_predicted', getStepStatus('model_predicted'))}
                </div>
                <div className="step-title">Model Predicted</div>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Results Section */}
      {result && !isLoading && (
        <div className="results-section">
          <h2>Prediction Result</h2>
          <p className="result-text">{result}</p>

          {uncertainty !== null && (
            <div className="uncertainty-box">
              <h3>Uncertainty</h3>
              <p>{(uncertainty * 100).toFixed(2)}%</p>
            </div>
          )}

          {probabilities.length > 0 && (
            <div className="probabilities-box">
              <h3>Class Probabilities</h3>
              {probabilities.map(([disease, prob], index) => (
                <div key={index} className="probability-bar">
                  <span>{disease}</span>
                  <div className="bar-container">
                    <div className="bar" style={{ width: `${prob * 100}%` }}></div>
                    <span className="probability-text">{(prob * 100).toFixed(2)}%</span>
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>
      )}
    </div>
  );
}

export default App;