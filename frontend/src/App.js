import React, { useState } from 'react';
import axios from 'axios';

function App() {
  // State for storing the uploaded file, image preview, prediction result, probabilities, heatmap URL, and uncertainty
  const [file, setFile] = useState(null);
  const [imagePreviewUrl, setImagePreviewUrl] = useState('');
  const [result, setResult] = useState('');
  const [probabilities, setProbabilities] = useState([]);
  const [uncertainty, setUncertainty] = useState(null);
  const [heatmapUrl, setHeatmapUrl] = useState(''); // State for storing heatmap URL

  // Handle file upload and preview
  const handleFileChange = (event) => {
    const file = event.target.files[0];
    setFile(file); // Store the file

    // Create image preview URL
    const reader = new FileReader();
    reader.onloadend = () => {
      setImagePreviewUrl(reader.result); // Set image preview URL
    };
    reader.readAsDataURL(file);
  };

  // Handle form submission to get predictions
  const handleSubmit = async () => {
    const formData = new FormData();
    formData.append('file', file); // Append the file to formData

    try {
      // Send request to Flask backend
      const response = await axios.post('http://localhost:5000/predict', formData, {
        headers: { 'Content-Type': 'multipart/form-data' },
      });

      const data = response.data;
      setResult(`Predicted Disease: ${data.disease_name}\nClass Index: ${data.class_index}`);
      setUncertainty(data.uncertainty); // Set uncertainty state from the response
      setProbabilities(Object.entries(data.class_probabilities)); // Store probabilities
      setHeatmapUrl(`http://localhost:5000${data.heatmap_url}`); // Set heatmap URL
    } catch (error) {
      console.error('Error:', error);
      setResult('Error occurred while predicting');
    }
  };

  return (
    <div className="App" style={{ height: '100vh', padding: '20px', backgroundColor: '#f4f4f4' }}>
      <h1 style={{ color: 'black' }}>Image Classifier</h1>

      {/* File upload and button */}
      <input type="file" onChange={handleFileChange} />
      <button onClick={handleSubmit}>Upload and Predict</button>

      {/* Display Image and Bar Chart side by side */}
      <div style={{ display: 'flex', width: '100%', marginTop: '30px' }}>
        
        {/* Image Preview Section */}
        {imagePreviewUrl && (
          <div style={{ width: '50%', padding: '20px' }}>
            <h3 style={{ color: 'black' }}>Uploaded Image</h3>
            <div style={{ width: '100%', height: '100%', borderRadius: '20px', overflow: 'hidden' }}>
              <img 
                src={imagePreviewUrl} 
                alt="Uploaded Preview" 
                style={{ width: '100%', height: '100%', objectFit: 'cover', borderRadius: '20px' }} 
              />
            </div>
          </div>
        )}

        {/* Bar Chart Section */}
        <div style={{ width: '50%', paddingLeft: '10px' }}>
          {uncertainty !== null && (
            <div style={{
              backgroundColor: '#282828',
              borderRadius: '20px',
              padding: '20px',
              color: 'white',
              boxShadow: '0 4px 8px rgba(0, 0, 0, 0.1)',
              marginBottom: '20px'
            }}>
              <h3>Uncertainty Quantification</h3>
              <p style={{ fontSize: '18px', fontWeight: 'bold' }}>Uncertainty: {uncertainty}%</p>
            </div>
          )}

          {probabilities.length > 0 && (
            <div style={{
              backgroundColor: '#282828',
              borderRadius: '20px',
              padding: '20px',
              color: 'white',
              boxShadow: '0 4px 8px rgba(0, 0, 0, 0.1)'
            }}>
              <h3>Class Probability Distribution</h3>

              {probabilities.map(([disease, prob], index) => (
                <div key={index} style={{ display: 'flex', alignItems: 'center', marginBottom: '20px' }}>
                  <div style={{ minWidth: '150px', textAlign: 'right', marginRight: '10px' }}>{disease}</div>
                  <div style={{
                    height: '10px',
                    width: `${prob * 100}%`,
                    backgroundColor: prob === Math.max(...probabilities.map(([_, p]) => p)) ? '#C6C7F8' : '#3E3E3E',
                    borderRadius: '10px',
                    paddingRight: '10px',
                    display: 'flex',
                    justifyContent: 'flex-end',
                    alignItems: 'center'
                  }}>
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>
      </div>

      {/* Heatmap Section */}
      {heatmapUrl && (
        <div style={{ marginTop: '80px', width: '100%', textAlign: 'center' }}>
          <h3>Heatmap</h3>
          <img 
            src={heatmapUrl} 
            alt="Prediction Heatmap"  
            style={{ width: '100%', borderRadius: '20px', boxShadow: '0 4px 8px rgba(0, 0, 0, 0.1)' }} 
          />
        </div>
      )}
    </div>
  );
}

export default App;