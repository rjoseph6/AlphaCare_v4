// src/components/Predictions.js

import React, { useEffect, useState, useRef } from 'react';
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

  

  // Zoom state
  const [scale, setScale] = useState(1);

  // Panning state
  const [translate, setTranslate] = useState({ x: 0, y: 0 });
  const [isDragging, setIsDragging] = useState(false);
  const lastPosition = useRef({ x: 0, y: 0 });

  // References
  const imageRef = useRef(null);

  // Zoom handlers
  const zoomIn = () => {
    setScale((prevScale) => Math.min(prevScale + 0.2, 3)); // Max scale 3
  };

  const zoomOut = () => {
    setScale((prevScale) => Math.max(prevScale - 0.2, 1)); // Min scale 1
    // Optionally reset translation when zooming out to default
    if (scale - 0.2 <= 1) {
      setTranslate({ x: 0, y: 0 });
    }
  };
  useEffect(() => {
    const zoomOutButton = document.querySelector('.zoom-out');
    if (zoomOutButton) {
      if (scale > 1) {
        zoomOutButton.classList.add('visible');
      } else {
        zoomOutButton.classList.remove('visible');
      }
    }
  }, [scale]);

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

  // Mouse event handlers for panning
  const handleMouseDown = (e) => {
    if (scale > 1) { // Enable dragging only when zoomed in
      setIsDragging(true);
      lastPosition.current = { x: e.clientX, y: e.clientY };
      e.preventDefault();
    }
  };

  const handleMouseMove = (e) => {
    if (isDragging) {
      const dx = e.clientX - lastPosition.current.x;
      const dy = e.clientY - lastPosition.current.y;
      setTranslate((prev) => ({ x: prev.x + dx, y: prev.y + dy }));
      lastPosition.current = { x: e.clientX, y: e.clientY };
    }
  };

  const handleMouseUp = () => {
    setIsDragging(false);
  };

  // Touch event handlers for mobile panning (optional)
  const handleTouchStart = (e) => {
    if (scale > 1) {
      setIsDragging(true);
      const touch = e.touches[0];
      lastPosition.current = { x: touch.clientX, y: touch.clientY };
    }
  };

  const handleTouchMove = (e) => {
    if (isDragging) {
      const touch = e.touches[0];
      const dx = touch.clientX - lastPosition.current.x;
      const dy = touch.clientY - lastPosition.current.y;
      setTranslate((prev) => ({ x: prev.x + dx, y: prev.y + dy }));
      lastPosition.current = { x: touch.clientX, y: touch.clientY };
    }
  };

  const handleTouchEnd = () => {
    setIsDragging(false);
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
              <div
                className="image-preview"
                onMouseDown={handleMouseDown}
                onMouseMove={handleMouseMove}
                onMouseUp={handleMouseUp}
                onMouseLeave={handleMouseUp}
                onTouchStart={handleTouchStart}
                onTouchMove={handleTouchMove}
                onTouchEnd={handleTouchEnd}
              >
                {/* Zoom Controls */}
                <div className="zoom-controls">
                  <button
                    className="zoom-button zoom-in"
                    onClick={zoomIn}
                    aria-label="Zoom In"
                    title="Zoom In"
                  >
                    +
                  </button>
                  {scale > 1 && (
                    <button
                      className="zoom-button zoom-out"
                      onClick={zoomOut}
                      aria-label="Zoom Out"
                      title="Zoom Out"
                    >
                      -
                    </button>
                  )}
                </div>
                {/* Image with Zoom and Pan */}
                <img
                  src={imagePreviewUrl}
                  alt="Uploaded Preview"
                  ref={imageRef}
                  style={{
                    transform: `scale(${scale}) translate(${translate.x}px, ${translate.y}px)`,
                    transition: isDragging ? 'none' : 'transform 0.2s ease-in-out',
                    cursor: isDragging ? 'grabbing' : 'grab',
                  }}
                />
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
                        ? `${uncertainty.toFixed(2)}%`
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