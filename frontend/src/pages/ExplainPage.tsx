import type { PredictResult } from "../types";

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || "http://127.0.0.1:8000";

function loadResult(): PredictResult | null {
  const raw = localStorage.getItem("latestDeepfakeResult");
  if (!raw) return null;
  try {
    return JSON.parse(raw) as PredictResult;
  } catch {
    return null;
  }
}

function toPublicUrl(path: string) {
  if (!path) return "";
  if (path.startsWith("http")) return path;
  if (path.startsWith("/outputs")) return `${API_BASE_URL}${path}`;
  return path;
}

function ExplainPage() {
  const result = loadResult();

  if (!result) {
    return (
      <section className="page-section">
        <div className="card">
          <h1>暂无可解释化结果</h1>
          <p className="hint">请先上传视频进行检测。</p>
        </div>
      </section>
    );
  }

  const fft = result.module_b_details?.fft;
  const spectrumUrl = fft?.spectrum_url ? toPublicUrl(fft.spectrum_url) : "";
  const llmText = result.llm_analysis?.text || result.explanation?.summary || "暂无 LLM 解释。";

  return (
    <section className="page-section explain-page-simple">
      <div className="page-header large-header">
        <p className="eyebrow">Explainability</p>
        <h1>频谱图与智能解释</h1>
        <p className="subtitle">该页面聚焦展示频谱可视化结果，以及基于本次 JSON 检测结果生成的自然语言解释。</p>
      </div>

      <div className="explain-simple-grid">
        <div className="card explain-visual-card">
          <div className="section-head-row">
            <div>
              <h2>FFT 频谱图</h2>
              <p className="hint">用于观察视频在频域上的整体特征分布。</p>
            </div>
          </div>

          {spectrumUrl ? (
            <div className="fft-image-wrap">
              <img className="fft-image" src={spectrumUrl} alt="FFT frequency spectrum" />
            </div>
          ) : (
            <div className="placeholder-box">当前结果暂无频谱图输出</div>
          )}

          <div className="metric-pills">
            <span>frequency_score：{fft?.frequency_score ?? "--"}</span>
            <span>FFT 高频均值：{fft?.fft_high_freq_ratio_mean ?? "--"}</span>
            <span>DCT 高频均值：{fft?.dct_high_freq_ratio_mean ?? "--"}</span>
          </div>
        </div>

        <div className="card explain-text-card">
          <div className="section-head-row">
            <div>
              <h2>智能取证解读</h2>
              <p className="hint">结合模型输出、频域证据和风险分数生成的综合分析说明。</p>
            </div>
          </div>

          <div className="llm-long-text">
            {llmText}
          </div>
        </div>
      </div>
    </section>
  );
}

export default ExplainPage;



