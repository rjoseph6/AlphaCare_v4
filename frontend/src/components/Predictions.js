// src/components/Predictions.js

import React, { useEffect } from 'react';
import { useLocation, useNavigate } from 'react-router-dom';
import './Predictions.css';

function Predictions() {
  const location = useLocation();
  const navigate = useNavigate();
  const {
    probabilities,
    uncertainty,
    error,
    imagePreviewUrl,
  } = location.state || {};

  // Redirect to home if there's no state
  useEffect(() => {
    if (!location.state) {
      navigate('/');
    }
  }, [location, navigate]);

  // Sort probabilities
  const sortedProbabilities = probabilities
    ? probabilities.sort((a, b) => b[1] - a[1])
    : [];

  // Function to map probabilities to descriptors
  const getProbabilityDescriptor = (prob) => {
    if (prob >= 0.8) return 'Very Likely';
    if (prob >= 0.6) return 'Likely';
    if (prob >= 0.4) return 'Possible';
    if (prob >= 0.2) return 'Unlikely';
    return 'Very Unlikely';
  };

  // Function to determine circle color based on probability
  const getCircleColor = (prob) => {
    if (prob >= 0.8) return '#28a745'; // Green for Very Likely
    if (prob >= 0.6) return '#ffc107'; // Yellow for Likely
    if (prob >= 0.4) return '#17a2b8'; // Blue for Possible
    if (prob >= 0.2) return '#dc3545'; // Red for Unlikely
    return '#6c757d'; // Grey for Very Unlikely
  };

  return (
    <div className="app-container">
      {/* Header */}
      <header className="app-header">
        <img src="logo_placeholder.svg" alt="Logo" className="logo" />
      </header>

      {/* Main Content */}
      <div className="main-content">
        <div className="predictions-container">
          {/* Left Panel */}
          <div className="left-panel">
            {imagePreviewUrl && (
              <div className="image-preview">
                <img src={imagePreviewUrl} alt="Uploaded Preview" />
              </div>
            )}
          </div>

          {/* Right Panel */}
          <div className="right-panel">
            {/* Display Error */}
            {error ? (
              <div className="error-message">
                <h2>Error</h2>
                <p>{error}</p>
              </div>
            ) : (
              <>
                {/* Top Two Boxes */}
                <div className="top-boxes">

                  {/* Most Predicted Class */}
                  <div className="box box-predicted">
                    <h2>{probabilities[0][0]}</h2>
                    <p className="percentage">
                      {probabilities && probabilities.length > 0
                        ? `${(probabilities[0][1] * 100).toFixed(2)}%`
                        : 'N/A'}
                    </p>
                  </div>

                  {/* Uncertainty Value */}
                  <div className="box box-uncertainty">
                    <h2>Uncertainty</h2>
                    <p className="percentage">
                      {uncertainty !== null
                        ? `${(uncertainty).toFixed(2)}%`
                        : 'N/A'}
                    </p>
                    {/* Optional: Add a progress bar */}
                    {uncertainty !== null && (
                      <div className="uncertainty-progress">
                        <div
                          className="uncertainty-bar"
                          style={{
                            width: `${((uncertainty / Math.log2(7)) * 100).toFixed(2)}%`,
                          }}
                        ></div>
                      </div>
                    )}
                  </div>
                </div>

                {/* Prediction List */}
                <div className="prediction-list">
                  {sortedProbabilities.map(([disease, prob], index) => (
                    <div key={index} className="prediction-item">
                      <span className="disease-name">{disease}</span>
                      <span className="probability-descriptor">
                        <span
                          className="probability-circle"
                          style={{ backgroundColor: getCircleColor(prob) }}
                        ></span>
                        {getProbabilityDescriptor(prob)}
                      </span>
                      <div className="probability-line-container">
                        {/* Dynamic Width Based on Probability */}
                        <div
                          className="probability-line"
                          style={{ width: `${prob * 100}%` }}
                        ></div>
                        {/* Circle Indicator Positioned at the End of the Line */}
                        <div
                          className="probability-circle-indicator"
                          style={{
                            left: `${prob * 100}%`,
                            
                          }}
                        ></div>
                      </div>
                    </div>
                  ))}
                </div>
              </>
            )}

            {/* Back Button */}
            <button className="back-button" onClick={() => navigate('/')}>
              Upload Another Image
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}

export default Predictions;