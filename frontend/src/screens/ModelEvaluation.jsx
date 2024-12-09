import { useEffect, useState } from "react";

function ModelEvaluation() {
  const [metrics, setMetrics] = useState(null);

  useEffect(() => {
    const fetchMetrics = async () => {
      const response = await fetch("http://localhost:8000/evaluate"); // Adjust this endpoint as needed
      const data = await response.json();
      setMetrics(data);
    };

    fetchMetrics();
  }, []);

  return (
    <div className="container">
      <h2>Model Evaluation Metrics</h2>
      {metrics ? (
        <div>
          <p>Accuracy: {metrics.accuracy}</p>
          <p>Precision: {metrics.precision}</p>
          <p>Recall: {metrics.recall}</p>
          {/* Add more metrics as needed */}
        </div>
      ) : (
        <p>Loading metrics...</p>
      )}
    </div>
  );
}

export default ModelEvaluation;
