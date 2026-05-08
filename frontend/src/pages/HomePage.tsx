import { Link } from "react-router-dom";

function HomePage() {
  return (
    <section className="landing-page">
      <div className="hero-grid hero-grid-large">
        <div className="hero-copy">
          <p className="eyebrow">Explainable Deepfake Detection Platform</p>
          <h1>
            DeepLens AI
            <br />
            让深度伪造检测
            <br />
            更直观、更可信
          </h1>
          <p className="subtitle hero-subtitle">
            这是一个面向视频 Deepfake 检测的可解释化平台。系统将视觉模型判断、频域分析与大模型自然语言解释融合在一起，帮助你不仅看到“结果”，更看到“为什么会得出这个结果”。
          </p>
          <div className="button-row">
            <Link className="primary-button" to="/upload">开始检测</Link>
            <Link className="secondary-button" to="/explain">查看可解释化</Link>
          </div>
        </div>

        <div className="hero-visual neural-panel">
          <div className="neural-bg" />
          <div className="hero-visual-main card glass-card">
            <div className="visual-chip">AI Forensics Engine</div>
            <div className="visual-screen">
              <div className="visual-grid" />
              <div className="visual-core" />
              <div className="visual-ring visual-ring-1" />
              <div className="visual-ring visual-ring-2" />
              <div className="visual-ring visual-ring-3" />
            </div>
            <div className="visual-stats">
              <div>
                <span>Visual</span>
                <b>EfficientNet</b>
              </div>
              <div>
                <span>Frequency</span>
                <b>FFT / DCT</b>
              </div>
              <div>
                <span>Explainability</span>
                <b>LLM Report</b>
              </div>
            </div>
          </div>
          <div className="floating-card floating-card-top">
            <strong>可解释检测</strong>
            <span>模型结果不止真假，还会附上证据链。</span>
          </div>
          <div className="floating-card floating-card-bottom">
            <strong>多模态辅助分析</strong>
            <span>视觉、频域、同步线索联合判断。</span>
          </div>
        </div>
      </div>

      <div className="feature-strip">
        <article className="card feature-card">
          <h3>上传检测</h3>
          <p>支持 mp4、mov、avi、mkv 视频，上传后自动调用后端模型。</p>
        </article>
        <article className="card feature-card">
          <h3>风险评分</h3>
          <p>输出真假判断、伪造概率、综合风险分数与风险等级。</p>
        </article>
        <article className="card feature-card">
          <h3>流程化解释</h3>
          <p>将检测流程拆解为视觉判断、频谱分析、LLM 解释等步骤逐一展示。</p>
        </article>
        <article className="card feature-card">
          <h3>频谱可视化</h3>
          <p>在可解释化页面直接展示频谱图，便于快速观察异常频域结构。</p>
        </article>
      </div>
    </section>
  );
}

export default HomePage;
