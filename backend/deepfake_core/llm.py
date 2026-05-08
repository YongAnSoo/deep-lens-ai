"""
LLM explanation layer.
Default provider: Zhipu GLM.
Output style: free-form natural explanation based on model JSON.
"""

import os
import json
from typing import Dict, Optional, List


DEFAULT_ZHIPU_MODEL = "glm-4-flash"
DEFAULT_ZHIPU_BASE_URL = "https://open.bigmodel.cn/api/paas/v4/"


def _make_json_safe(result: Dict) -> Dict:
    """
    Keep the most useful information for LLM explanation.
    Very long arrays are shortened to avoid making the prompt noisy.
    """
    safe = dict(result)

    try:
        module_b_details = safe.get("module_b_details") or {}
        fft = module_b_details.get("fft") or {}

        if "fft_ratios" in fft and isinstance(fft["fft_ratios"], list):
            fft["fft_ratios_preview"] = fft["fft_ratios"][:8]
            fft["fft_ratios_count"] = len(fft["fft_ratios"])
            fft.pop("fft_ratios", None)

        if "dct_ratios" in fft and isinstance(fft["dct_ratios"], list):
            fft["dct_ratios_preview"] = fft["dct_ratios"][:8]
            fft["dct_ratios_count"] = len(fft["dct_ratios"])
            fft.pop("dct_ratios", None)

        module_b_details["fft"] = fft
        safe["module_b_details"] = module_b_details
    except Exception:
        pass

    return safe


def build_explanation_prompt(
    result: Dict,
    *,
    suspicious_frames: Optional[List[int]] = None,
    gradcam_paths: Optional[Dict[int, Dict[str, str]]] = None,
    frequency_summary: Optional[Dict] = None,
    sync_summary: Optional[Dict] = None,
) -> str:
    safe_result = _make_json_safe(result)

    json_text = json.dumps(
        safe_result,
        ensure_ascii=False,
        indent=2,
        default=str,
    )

    parts = [
        "你是一名深度伪造检测与多媒体取证分析助手。",
        "",
        "下面是一段 Web Demo 后端返回的模型检测 JSON。请你根据 JSON 中的内容，自由生成一段自然、专业、易懂的中文解释。",
        "",
        "输出要求：",
        "1. 不要使用固定模板，不要强制使用【最终判断】【主要证据】等标题。",
        "2. 不要写成机械的项目符号列表，尽量像一段完整的分析说明。",
        "3. 可以自由组织语言，但要结合 JSON 里的真假判断、伪造概率、风险分数、风险等级、人脸数量、FFT 频域结果、Sync 音视频同步结果、Grad-CAM 或可疑帧信息。",
        "4. 如果模型判断为真实或低风险，不要说“疑似深度伪造”，要明确说明当前证据更偏向真实/低风险，但仍然属于模型辅助判断。",
        "5. 如果模型判断为伪造或高风险，要说明哪些证据支持这个判断。",
        "6. 如果 Grad-CAM 为空，要说明当前没有可疑帧热力图证据，不要编造热力图内容。",
        "7. 不要声称结果是绝对真实或绝对伪造，要保留不确定性。",
        "8. 输出长度建议 150 到 300 字。",
        "",
        "模型检测 JSON：",
        json_text,
    ]

    return "\n".join(parts)


def _fallback_report(
    result: Dict,
    *,
    suspicious_frames: Optional[List[int]] = None,
    gradcam_paths: Optional[Dict[int, Dict[str, str]]] = None,
    frequency_summary: Optional[Dict] = None,
    sync_summary: Optional[Dict] = None,
) -> Dict:
    label = result.get("video_label", "unknown")
    visual_prob = float(result.get("video_fake_prob", 0.5))
    risk_score = result.get("risk_score")
    severity = result.get("severity", "unknown")
    num_faces = result.get("num_faces", 0)

    module_b = result.get("module_b") or {}
    freq = module_b.get("frequency_score")
    sync = module_b.get("sync_score")

    text = (
        f"模型本次将视频判断为 {label}，伪造概率约为 {visual_prob:.3f}，"
        f"综合风险分数为 {risk_score}，风险等级为 {severity}。"
        f"检测过程中共识别到 {num_faces} 张人脸。"
    )

    if freq is not None:
        text += f" 频域分析分数为 {freq}，可作为判断视频压缩痕迹或异常频率成分的辅助证据。"

    if sync is not None:
        text += f" 音视频同步风险分数为 {sync}，用于辅助观察画面与声音之间是否存在不同步现象。"

    if gradcam_paths:
        text += " 同时系统生成了 Grad-CAM 热力图，可用于观察模型关注的画面区域。"
    else:
        text += " 当前没有生成 Grad-CAM 热力图，因此不能从热力图角度进一步解释模型关注区域。"

    text += " 该结果应作为自动化辅助分析，而不是绝对鉴定结论，最终仍建议结合视频来源、人工复核和其他检测方法综合判断。"

    return {
        "provider": "local_template",
        "model": "none",
        "text": text,
    }


def _call_zhipu_api(
    *,
    api_key: str,
    base_url: str,
    model: str,
    prompt: str,
) -> str:
    from openai import OpenAI

    client = OpenAI(
        api_key=api_key,
        base_url=base_url,
    )

    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": "你是一名专业但表达自然的中文深度伪造检测解释助手。",
            },
            {
                "role": "user",
                "content": prompt,
            },
        ],
        temperature=0.45,
    )

    if response.choices:
        return response.choices[0].message.content.strip()

    return ""


def generate_llm_explanation(
    result: Dict,
    model: str = DEFAULT_ZHIPU_MODEL,
    *,
    provider: Optional[str] = None,
    suspicious_frames: Optional[List[int]] = None,
    gradcam_paths: Optional[Dict[int, Dict[str, str]]] = None,
    frequency_summary: Optional[Dict] = None,
    sync_summary: Optional[Dict] = None,
) -> Dict:
    prompt = build_explanation_prompt(
        result,
        suspicious_frames=suspicious_frames,
        gradcam_paths=gradcam_paths,
        frequency_summary=frequency_summary,
        sync_summary=sync_summary,
    )

    provider = (provider or os.getenv("LLM_PROVIDER") or "zhipu").strip().lower()

    if provider not in ["zhipu", "glm", "bigmodel"]:
        provider = "zhipu"

    api_key = os.getenv("ZHIPU_API_KEY") or os.getenv("BIGMODEL_API_KEY")

    if not api_key:
        return _fallback_report(
            result,
            suspicious_frames=suspicious_frames,
            gradcam_paths=gradcam_paths,
            frequency_summary=frequency_summary,
            sync_summary=sync_summary,
        )

    base_url = os.getenv("ZHIPU_BASE_URL", DEFAULT_ZHIPU_BASE_URL)
    chosen_model = os.getenv("ZHIPU_MODEL", model or DEFAULT_ZHIPU_MODEL)

    try:
        text = _call_zhipu_api(
            api_key=api_key,
            base_url=base_url,
            model=chosen_model,
            prompt=prompt,
        )

        if not text:
            return _fallback_report(
                result,
                suspicious_frames=suspicious_frames,
                gradcam_paths=gradcam_paths,
                frequency_summary=frequency_summary,
                sync_summary=sync_summary,
            )

        return {
            "provider": "zhipu",
            "model": chosen_model,
            "base_url": base_url,
            "text": text,
        }

    except Exception as e:
        fallback = _fallback_report(
            result,
            suspicious_frames=suspicious_frames,
            gradcam_paths=gradcam_paths,
            frequency_summary=frequency_summary,
            sync_summary=sync_summary,
        )
        fallback["error"] = str(e)
        return fallback
