import { useState } from "react";

function ModelRetrain() {
  const [file, setFile] = useState(null);
  const [message, setMessage] = useState("");

  const handleFileChange = (e) => {
    setFile(e.target.files[0]);
  };

  const handleUpload = async () => {
    const formData = new FormData();
    formData.append("file", file);

    const response = await fetch("http://localhost:8000/retrain", {
      method: "POST",
      body: formData,
    });

    const data = await response.json();
    setMessage(data.message);
  };

  return (
    <div className="container">
      <h2>Upload Data for Model Retraining</h2>
      <input type="file" accept=".csv" onChange={handleFileChange} />
      <button onClick={handleUpload}>Retrain</button>
      {message && <p>{message}</p>}
    </div>
  );
}

export default ModelRetrain;
