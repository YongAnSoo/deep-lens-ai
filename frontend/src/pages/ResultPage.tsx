import { Link } from "react-router-dom";
import type { PredictResult } from "../types";

function loadResult(): PredictResult | null {
  const raw = localStorage.getItem("latestDeepfakeResult");
  if (!raw) return null;
  try {
    return JSON.parse(raw) as PredictResult;
  } catch {
    return null;
  }
}

function percent(value?: number | null) {
  if (value === null || value === undefined || Number.isNaN(value)) return "--";
  return `${(value * 100).toFixed(2)}%`;
}

function getExplanation(result: PredictResult) {
  return result.llm_analysis?.text || result.explanation?.summary || "暂无解释。";
}

function ResultPage() {
  const result = loadResult();

  if (!result) {
    return (
      <section className="page-section">
        <div className="card">
          <h1>暂无检测结果</h1>
          <p className="hint">请先上传一个视频进行检测。</p>
          <Link className="primary-button" to="/upload">前往上传</Link>
        </div>
      </section>
    );
  }

  const isFake = result.label === "fake";
  const verdict = isFake ? "疑似 Deepfake" : result.label === "real" ? "疑似真实" : "未知";
  const riskScore = typeof result.risk_score === "number" ? result.risk_score.toFixed(2) : "--";
  const fft = result.module_b_details?.fft;
  const sync = result.module_b_details?.sync;

  const flowSteps = [
    {
      title: "视频输入",
      desc: `系统接收文件 ${result.filename}，并建立本次任务的检测记录。`,
      value: result.task_id,
    },
    {
      title: "视觉模型检测",
      desc: "后端对抽取到的人脸帧进行视觉分类，输出伪造概率与真假判断。",
      value: `伪造概率 ${percent(result.fake_probability)}`,
    },
    {
      title: "频域证据分析",
      desc: "平台进一步分析视频频谱特征，用于观察是否存在异常高频模式。",
      value: fft?.frequency_score !== undefined ? `frequency_score ${fft.frequency_score}` : "无额外频域分数",
    },
    {
      title: "同步辅助分析",
      desc: "如启用同步模块，将给出音视频时序相关性的辅助判断。",
      value: sync?.sync_score !== undefined ? `sync_score ${sync.sync_score}` : "未提供同步分数",
    },
    {
      title: "综合风险评估",
      desc: "系统融合多项证据后，得到最终风险分数与风险等级。",
      value: `风险分数 ${riskScore} / 风险等级 ${result.severity || "--"}`,
    },
    {
      title: "自然语言解释",
      desc: "LLM 将检测证据整合为更易理解的分析结果。",
      value: `来源 ${result.llm_analysis?.provider || "local_template"}`,
    },
  ];

  return (
    <section className="page-section result-page">
      <div className="page-header large-header">
        <p className="eyebrow">Result</p>
        <h1>检测结果</h1>
        <p className="subtitle">以下内容按照检测流程逐步展示本次视频分析是如何完成的。</p>
      </div>

      <div className="summary-grid">
        <div className="summary-card emphasis-card">
          <span>最终判断</span>
          <strong className={isFake ? "fake-label" : "real-label"}>{verdict}</strong>
        </div>
        <div className="summary-card">
          <span>伪造概率</span>
          <strong>{percent(result.fake_probability)}</strong>
        </div>
        <div className="summary-card">
          <span>风险分数</span>
          <strong>{riskScore}</strong>
        </div>
        <div className="summary-card">
          <span>风险等级</span>
          <strong>{result.severity || "--"}</strong>
        </div>
      </div>

      <div className="card flow-wrapper">
        <div className="flow-header">
          <h2>检测流程解释</h2>
          <p className="hint">从输入、分析到结论，每一步都对应本次结果中的一项证据。</p>
        </div>

        <div className="flow-list">
          {flowSteps.map((step, index) => (
            <div className="flow-step" key={step.title}>
              <div className="flow-index">{String(index + 1).padStart(2, "0")}</div>
              <div className="flow-content">
                <h3>{step.title}</h3>
                <p>{step.desc}</p>
                <div className="flow-badge">{step.value}</div>
              </div>
            </div>
          ))}
        </div>
      </div>

      <div className="card narrative-box">
        <h2>综合解释</h2>
        <p>{getExplanation(result)}</p>
      </div>

      <div className="button-row">
        <Link className="secondary-button" to="/explain">查看频谱图与 LLM 解释</Link>
        <Link className="primary-button" to="/upload">继续检测</Link>
      </div>
    </section>
  );
}

export default ResultPage;
