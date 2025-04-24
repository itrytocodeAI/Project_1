import React, { useRef, useState } from "react";
import axios from "axios";
import "./index.css";

function App() {
  const fileInputRef = useRef(null);
  const [file, setFile] = useState(null);
  const [preview, setPreview] = useState(null);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);

  const handleUploadClick = () => {
    fileInputRef.current.click();
  };

  const handleFileChange = (e) => {
    const selectedFile = e.target.files[0];
    setFile(selectedFile);
    setResult(null);
    if (selectedFile) {
      setPreview(URL.createObjectURL(selectedFile));
    }
  };

  const handleDescribe = async () => {
    if (!file) return alert("Please upload an image first.");
    const formData = new FormData();
    formData.append("file", file);

    setLoading(true);
    try {
      const response = await axios.post("http://localhost:8000/predict", formData);
      setResult(response.data);
    } catch (error) {
      alert("Failed to get description: " + (error.response?.data?.error || error.message));
    }
    setLoading(false);
  };

  return (
    
    <div className="app">
    <nav>AI vs Real Detector</nav>
      <h1>AI vs Real Image Detector</h1>

      {/* Hidden file input */}
      <input
        type="file"
        accept="image/*"
        ref={fileInputRef}
        onChange={handleFileChange}
        style={{ display: "none" }}
      />

      {/* Upload button triggers file input */}
      <button onClick={handleUploadClick}>Upload Image</button>

      {preview && (
        <div className="preview">
          <img
  src={preview}
  alt="Preview"
  style={{
    marginTop: "1rem",
    width: "240px",
    height: "240px",
    objectFit: "cover",       // Crop if needed
    borderRadius: "10px",
    boxShadow: "0 4px 10px rgba(0,0,0,0.15)"
  }}
/>      
        </div>
      )}

      <button onClick={handleDescribe} disabled={!file || loading} style={{ marginTop: "1rem" }}>
        {loading ? "Describing..." : "Describe Image"}
      </button>

      {result && (
        <div className="result">
          <p><strong>Description:</strong> {result.description}</p>
          <p><strong>Image Type:</strong> {result.is_real ? "Real" : "Fake"}</p>
        </div>
      )}
      
    </div>
  );
}

export default App;
