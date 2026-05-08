import { useState } from "react";
import { Link, Route, Routes } from "react-router-dom";
import HomePage from "./pages/HomePage";
import UploadPage from "./pages/UploadPage";
import ResultPage from "./pages/ResultPage";
import ExplainPage from "./pages/ExplainPage";
import type { DetectionResult } from "./types";

export default function App() {
  const [result, setResultState] = useState<DetectionResult | null>(() => {
    const saved = localStorage.getItem("deeplens_result");

    if (!saved) {
      return null;
    }

    try {
      return JSON.parse(saved) as DetectionResult;
    } catch {
      localStorage.removeItem("deeplens_result");
      return null;
    }
  });

  const setResult = (value: DetectionResult | null) => {
    setResultState(value);

    if (value) {
      localStorage.setItem("deeplens_result", JSON.stringify(value));
    } else {
      localStorage.removeItem("deeplens_result");
    }
  };

  return (
    <div className="app-shell">
      <nav className="navbar">
        <Link className="brand" to="/">
          <span className="brand-mark">DL</span>
          <span>DeepLens AI</span>
        </Link>

        <div className="nav-links">
          <Link to="/">首页</Link>
          <Link to="/upload">上传检测</Link>
          <Link to="/result">结果页</Link>
          <Link to="/explain">可解释化</Link>
        </div>
      </nav>

      <Routes>
        <Route path="/" element={<HomePage />} />
        <Route path="/upload" element={<UploadPage setResult={setResult} />} />
        <Route path="/result" element={<ResultPage result={result} />} />
        <Route path="/explain" element={<ExplainPage result={result} />} />
      </Routes>
    </div>
  );
}
