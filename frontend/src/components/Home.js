// src/components/Home.js

import React, { useState } from 'react'; // Import React and useState
import { useNavigate } from 'react-router-dom';
import axios from 'axios';
import './Home.css'; // Import Home-specific CSS
import logo from '../assets/logo.svg'; // Import your logo if used

function Home() {
  // State variables
  const [file, setFile] = useState(null); // If 'file' is not used elsewhere, you can remove this
  const [imagePreviewUrl, setImagePreviewUrl] = useState('');
  const [currentStage, setCurrentStage] = useState('');
  const [isLoading, setIsLoading] = useState(false);

  const navigate = useNavigate();

  // Handle file upload and preview
  const handleFileChange = (event) => {
    event.preventDefault();
    let selectedFile;

    if (event.dataTransfer) {
      selectedFile = event.dataTransfer.files[0];
    } else {
      selectedFile = event.target.files[0];
    }

    if (!selectedFile) return;

    // setFile(selectedFile); // Uncomment if 'file' is used elsewhere

    // Create image preview URL
    const reader = new FileReader();
    reader.onloadend = async () => {
      const imageUrl = reader.result;
      setImagePreviewUrl(imageUrl);

      // Reset current stage
      setCurrentStage('image_uploaded');

      // Automatically submit the image for prediction
      await handleSubmit(selectedFile, imageUrl);
    };
    reader.readAsDataURL(selectedFile);
  };

  // Handle form submission to get predictions
  const handleSubmit = async (selectedFile, imageUrl) => {
    const formData = new FormData();
    formData.append('file', selectedFile);

    setIsLoading(true);
    setCurrentStage('skin');

    try {
      // Send request to backend for prediction
      const response = await axios.post(
        'http://localhost:5000/predict',
        formData,
        {
          headers: { 'Content-Type': 'multipart/form-data' },
        }
      );

      const data = response.data;
      setCurrentStage(data.stage);

      // Navigate to the predictions page with state
      navigate('/predictions', {
        state: {
          result: data.disease_name,
          probabilities: Object.entries(data.class_probabilities),
          uncertainty: data.uncertainty,
          imagePreviewUrl: imageUrl,
        },
      });
    } catch (error) {
      console.error('Error:', error);
      setCurrentStage('error');

      // Navigate to the predictions page with error state
      navigate('/predictions', {
        state: {
          error: 'Error occurred while predicting',
          imagePreviewUrl: imageUrl,
        },
      });
    } finally {
      setIsLoading(false);
    }
  };

  // Optionally, you can remove 'file', 'currentStage', and 'isLoading' if they're not used in the JSX

  return (
    <div className="app-container">
      {/* Header with Logo */}
      <header className="home-header">
        <img src={logo} alt="Logo" className="logo" />
      </header>

      {/* Main Content */}
      <div className="home-main-content">
        <h1 className="home-title">AlphaCare+</h1>

        {/* Drag and Drop Area */}
        <div
          className="home-upload-area"
          onClick={() => document.getElementById('fileInput').click()}
          onDragOver={(e) => e.preventDefault()}
          onDrop={handleFileChange}
        >
          <div className="upload-icon"></div> {/* Add an icon or image if desired */}
          <p className="upload-text">
            Drop your image here
            <span className="upload-subtext">Supports: JPG, PNG, DICOM</span>
          </p>
          <input
            id="fileInput"
            type="file"
            accept="image/*"
            onChange={handleFileChange}
            style={{ display: 'none' }}
          />
        </div>

        {/* Image Preview and Stepper */}
        {imagePreviewUrl && (
          <div className="content-section">
            {/* Image Preview */}
            <div className="image-preview">
              <h3>Uploaded Image</h3>
              <img src={imagePreviewUrl} alt="Uploaded Preview" />
            </div>

            {/* Stepper Visualization */}
            {/* Include your stepper visualization here if needed */}
          </div>
        )}
      </div>

      {/* Disclaimer */}
      <div className="disclaimer">
        AlphaCare+ is a prototype.{' '}
        <a href="terms-of-service.html">Read our Terms of Service</a>.
      </div>
    </div>
  );
}

export default Home;