import { BrowserRouter as Router, Route, Routes, Link } from "react-router-dom";
import Home from "./screens/Home";
import DataUpload from "./screens/DataUpload";
import ModelRetrain from "./screens/ModelRetrain";
import ModelEvaluation from "./screens/ModelEvaluation";
import Prediction from "./screens/Prediction";
import "./App.css";

function App() {
  return (
    <Router>
      <nav>
        <ul>
          <li>
            <Link to="/">Home</Link>
          </li>
          <li>
            <Link to="/upload">Upload Data</Link>
          </li>
          <li>
            <Link to="/retrain">Model Retrain</Link>
          </li>
          <li>
            <Link to="/evaluation">Model Evaluation</Link>
          </li>
          <li>
            <Link to="/predict">Make Prediction</Link>
          </li>
        </ul>
      </nav>
      <Routes>
        <Route path="/" element={<Home />} />
        <Route path="/upload" element={<DataUpload />} />
        <Route path="/retrain" element={<ModelRetrain />} />
        <Route path="/evaluation" element={<ModelEvaluation />} />
        <Route path="/predict" element={<Prediction />} />
      </Routes>
    </Router>
  );
}

export default App;
