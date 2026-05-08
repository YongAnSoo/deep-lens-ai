import { useState } from "react";
import { useNavigate } from "react-router-dom";
import type { PredictResult } from "../types";

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || "http://127.0.0.1:8000";

function UploadPage() {
  const navigate = useNavigate();
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [loading, setLoading] = useState(false);
  const [errorMessage, setErrorMessage] = useState("");

  const handlePredict = async () => {
    if (!selectedFile) {
      setErrorMessage("请先选择一个视频文件。");
      return;
    }

    setLoading(true);
    setErrorMessage("");

    try {
      const formData = new FormData();
      formData.append("file", selectedFile);

      const response = await fetch(`${API_BASE_URL}/predict`, {
        method: "POST",
        body: formData,
      });

      const data: PredictResult & { detail?: string } = await response.json();

      if (!response.ok) {
        throw new Error(data.detail || "检测失败");
      }

      localStorage.setItem("latestDeepfakeResult", JSON.stringify(data));
      navigate("/result");
    } catch (error) {
      setErrorMessage(error instanceof Error ? error.message : "发生未知错误");
    } finally {
      setLoading(false);
    }
  };

  return (
    <section className="upload-page">
      <div className="hero-grid upload-hero-grid">
        <div className="hero-copy">
          <p className="eyebrow">AI Video Forensics</p>
          <h1>
            上传视频，
            <br />
            一键获取深伪检测结果
          </h1>
          <p className="subtitle hero-subtitle">
            平台将自动完成视频接收、人脸帧分析、频域证据提取与大模型解释生成。建议上传 30 秒以内、画面较清晰、人物面部相对明显的视频，以获得更稳定的检测结果。
          </p>

          <div className="hero-highlights">
            <div className="mini-pill">视觉模型判断</div>
            <div className="mini-pill">频谱图输出</div>
            <div className="mini-pill">LLM 智能解释</div>
          </div>
        </div>

        <div className="hero-visual neural-panel compact-panel">
          <div className="neural-bg" />
          <div className="hero-visual-main card glass-card upload-preview-card">
            <div className="upload-preview-top">
              <span className="visual-chip">Detection Pipeline</span>
              <span className="status-dot">Ready</span>
            </div>
            <div className="pipeline-preview">
              <div className="pipeline-node active">Upload</div>
              <div className="pipeline-line" />
              <div className="pipeline-node">Model</div>
              <div className="pipeline-line" />
              <div className="pipeline-node">FFT</div>
              <div className="pipeline-line" />
              <div className="pipeline-node">LLM</div>
            </div>
            <div className="mock-spectrum" />
          </div>
        </div>
      </div>

      <div className="card upload-panel">
        <div className="upload-panel-head">
          <div>
            <p className="eyebrow">Upload</p>
            <h2>选择待检测视频</h2>
            <p className="hint">支持 mp4、mov、avi、mkv 格式。</p>
          </div>
        </div>

        <label className="upload-dropzone">
          <input
            className="file-input-hidden"
            type="file"
            accept=".mp4,.mov,.avi,.mkv"
            onChange={(event) => {
              setSelectedFile(event.target.files?.[0] || null);
              setErrorMessage("");
            }}
          />
          <div className="upload-dropzone-inner">
            <div className="upload-icon">↑</div>
            <h3>点击选择视频文件</h3>
            <p>将待检测视频上传到系统，随后自动开始模型分析流程。</p>
            {selectedFile && <p className="selected-file">已选择：{selectedFile.name}</p>}
          </div>
        </label>

        <div className="button-row upload-actions">
          <button className="primary-button" onClick={handlePredict} disabled={loading}>
            {loading ? "检测中，请稍候..." : "开始检测"}
          </button>
        </div>

        {errorMessage && <p className="error-text">{errorMessage}</p>}
      </div>
    </section>
  );
}

export default UploadPage;
