import React, { useState } from "react";
import "./Prediction.css"; // Create a separate CSS file for Prediction styles

function Prediction() {
  const [formData, setFormData] = useState({
    age: "",
    sex: "",
    cp: "",
    trestbps: "",
    chol: "",
    fbs: "",
    restecg: "",
    thalch: "",
    exang: "",
    oldpeak: "",
    slope: "",
    ca: "",
    thal: "",
  });

  const [prediction, setPrediction] = useState(null);

  const handleChange = (e) => {
    const { name, value } = e.target;
    setFormData({
      ...formData,
      [name]: value,
    });
  };

  const handlePredict = async () => {
    try {
      const response = await fetch("http://localhost:8000/predict", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify(formData),
      });

      if (!response.ok) {
        throw new Error("Prediction request failed");
      }

      const data = await response.json();
      setPrediction(data.Prediction);
    } catch (error) {
      console.error("Error:", error);
      alert("An error occurred while making the prediction. Please try again.");
    }
  };

  return (
    <div className="container">
      <h2>Make a Prediction</h2>
      <div className="form-group">
        <label>Age:</label>
        <input
          type="number"
          name="age"
          placeholder="Age"
          value={formData.age}
          onChange={handleChange}
          min="0"
        />
      </div>
      <div className="form-group">
        <label>Sex:</label>
        <select name="sex" value={formData.sex} onChange={handleChange}>
          <option value="">Select Sex</option>
          <option value="male">Male</option>
          <option value="female">Female</option>
        </select>
      </div>
      <div className="form-group">
        <label>Chest Pain Type:</label>
        <select name="cp" value={formData.cp} onChange={handleChange}>
          <option value="">Select Chest Pain Type</option>
          <option value="typical angina">Typical Angina</option>
          <option value="atypical angina">Atypical Angina</option>
          <option value="asymptomatic">Asymptomatic</option>
          <option value="non-anginal">Non-Anginal</option>
        </select>
      </div>
      <div className="form-group">
        <label>Resting Blood Pressure:</label>
        <input
          type="number"
          name="trestbps"
          placeholder="Resting Blood Pressure"
          value={formData.trestbps}
          onChange={handleChange}
          min="0"
        />
      </div>
      <div className="form-group">
        <label>Cholesterol:</label>
        <input
          type="number"
          name="chol"
          placeholder="Cholesterol"
          value={formData.chol}
          onChange={handleChange}
          min="0"
        />
      </div>
      <div className="form-group">
        <label>Fasting Blood Sugar:</label>
        <select name="fbs" value={formData.fbs} onChange={handleChange}>
          <option value="">Select Fasting Blood Sugar</option>
          <option value="0">False</option>
          <option value="1">True</option>
        </select>
      </div>
      <div className="form-group">
        <label>Resting ECG:</label>
        <select name="restecg" value={formData.restecg} onChange={handleChange}>
          <option value="">Select Resting ECG</option>
          <option value="lv hypertrophy">LV Hypertrophy</option>
          <option value="normal">Normal</option>
          <option value="st-t abnormality">ST-T Abnormality</option>
        </select>
      </div>
      <div className="form-group">
        <label>Maximum Heart Rate Achieved:</label>
        <input
          type="number"
          name="thalch"
          placeholder="Maximum Heart Rate Achieved"
          value={formData.thalch}
          onChange={handleChange}
          min="0"
        />
      </div>
      <div className="form-group">
        <label>Exercise Induced Angina:</label>
        <select name="exang" value={formData.exang} onChange={handleChange}>
          <option value="">Select Exercise Induced Angina</option>
          <option value="0">False</option>
          <option value="1">True</option>
        </select>
      </div>
      <div className="form-group">
        <label>Oldpeak:</label>
        <input
          type="number"
          step="0.1"
          name="oldpeak"
          placeholder="Oldpeak"
          value={formData.oldpeak}
          onChange={handleChange}
          min="0"
        />
      </div>
      <div className="form-group">
        <label>Slope:</label>
        <select name="slope" value={formData.slope} onChange={handleChange}>
          <option value="">Select Slope</option>
          <option value="flat">Flat</option>
          <option value="downsloping">Downsloping</option>
          <option value="upsloping">Upsloping</option>
        </select>
      </div>
      <div className="form-group">
        <label>Number of Major Vessels:</label>
        <input
          type="number"
          name="ca"
          placeholder="Number of Major Vessels"
          value={formData.ca}
          onChange={handleChange}
          min="0"
        />
      </div>
      <div className="form-group">
        <label>Thalassemia:</label>
        <select name="thal" value={formData.thal} onChange={handleChange}>
          <option value="">Select Thalassemia</option>
          <option value="fixed defect">Fixed Defect</option>
          <option value="reversable defect">Reversable Defect</option>
          <option value="normal">Normal</option>
        </select>
      </div>
      <button onClick={handlePredict}>Predict</button>
      {prediction && <p>Prediction: {prediction}</p>}
    </div>
  );
}

export default Prediction;
