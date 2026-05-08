"""
视频级风险融合逻辑
将多个人脸预测结果融合为视频级风险评分
"""

import logging
from typing import List, Dict, Tuple
import numpy as np

from . import config


def calculate_video_risk(
    face_predictions: List[Dict[str, float]],
    method: str = "top_vote"
) -> Dict[str, float]:
    """
    计算视频级风险评分

    Args:
        face_predictions: 人脸预测列表，每个元素包含 {'real_prob', 'fake_prob', 'predicted_label'}
        method: 融合方法
            - "average": 平均所有人脸的 fake 概率
            - "max": 取最大的 fake 概率
            - "top_vote": 取 top-k 最高 fake 概率的投票
            - "weighted_average": 加权平均（置信度加权）

    Returns:
        {
            'video_fake_prob': 视频级 fake 概率,
            'video_real_prob': 视频级 real 概率,
            'video_label': 视频级标签 ('real' or 'fake'),
            'num_faces': 人脸数量,
            'method': 融合方法
        }
    """
    if not face_predictions:
        logging.warning("没有人脸预测结果，返回默认值")
        return {
            'video_fake_prob': 0.5,
            'video_real_prob': 0.5,
            'video_label': 'unknown',
            'num_faces': 0,
            'method': method
        }

    num_faces = len(face_predictions)
    fake_probs = [pred['fake_prob'] for pred in face_predictions]
    real_probs = [pred['real_prob'] for pred in face_predictions]

    # 根据方法计算视频级概率
    vote_details = None

    if method == "average":
        video_fake_prob = np.mean(fake_probs)
        video_real_prob = np.mean(real_probs)

    elif method == "max":
        video_fake_prob = np.max(fake_probs)
        video_real_prob = 1.0 - video_fake_prob

    elif method == "top_vote":
        # 取 top-k 最高 fake 概率的人脸进行投票
        k = max(1, int(num_faces * config.TOP_VOTE_PERCENT))
        top_k_indices = np.argsort(fake_probs)[-k:]
        top_k_labels = [face_predictions[i]['predicted_label'] for i in top_k_indices]
        top_k_fake_probs = [face_predictions[i]['fake_prob'] for i in top_k_indices]

        # 统计投票
        fake_votes = sum(1 for label in top_k_labels if label == 'fake')
        real_votes = k - fake_votes

        vote_details = {
            'k': int(k),
            'top_k_indices': [int(i) for i in top_k_indices],
            'fake_votes': int(fake_votes),
            'real_votes': int(real_votes),
            'top_k_fake_prob_mean': float(np.mean(top_k_fake_probs)),
            'top_k_fake_prob_min': float(np.min(top_k_fake_probs)),
            'top_k_fake_prob_max': float(np.max(top_k_fake_probs)),
        }

        # 计算概率（基于投票比例）
        video_fake_prob = fake_votes / k
        video_real_prob = real_votes / k

    elif method == "weighted_average":
        # 使用置信度作为权重
        weights = [max(pred['fake_prob'], pred['real_prob']) for pred in face_predictions]
        total_weight = sum(weights)

        if total_weight > 0:
            video_fake_prob = sum(w * p for w, p in zip(weights, fake_probs)) / total_weight
            video_real_prob = sum(w * p for w, p in zip(weights, real_probs)) / total_weight
        else:
            video_fake_prob = np.mean(fake_probs)
            video_real_prob = np.mean(real_probs)

    else:
        raise ValueError(f"不支持的融合方法: {method}")

    # 归一化概率
    total_prob = video_fake_prob + video_real_prob
    if total_prob > 0:
        video_fake_prob /= total_prob
        video_real_prob /= total_prob

    # 确定视频级标签
    video_label = 'fake' if video_fake_prob >= config.THRESHOLD else 'real'

    result = {
        'video_fake_prob': float(video_fake_prob),
        'video_real_prob': float(video_real_prob),
        'video_label': video_label,
        'num_faces': num_faces,
        'method': method
    }

    if vote_details is not None:
        result['vote_details'] = vote_details

    return result


def calculate_risk_score(
    video_fake_prob: float,
    num_faces: int,
    face_predictions: List[Dict[str, float]] = None
) -> Dict[str, float]:
    """
    计算详细的风险评分

    Args:
        video_fake_prob: 视频级 fake 概率
        num_faces: 人脸数量
        face_predictions: 人脸预测列表（可选，用于计算额外指标）

    Returns:
        {
            'risk_score': 综合风险评分 (0-100),
            'confidence': 置信度 (0-1),
            'consistency': 一致性 (0-1),
            'severity': 严重程度 ('low', 'medium', 'high')
        }
    """
    # 基础风险评分（基于 fake 概率）
    risk_score = video_fake_prob * 100

    # 计算置信度（基于概率的确定性）
    confidence = abs(video_fake_prob - 0.5) * 2  # 0.5 -> 0, 1.0 -> 1

    # 计算一致性（如果有人脸预测）
    if face_predictions and len(face_predictions) > 1:
        fake_probs = [pred['fake_prob'] for pred in face_predictions]
        consistency = 1.0 - np.std(fake_probs)  # 标准差越小，一致性越高
        consistency = max(0.0, min(1.0, consistency))  # 限制在 [0, 1]
    else:
        consistency = 1.0  # 单个人脸默认一致性为 1

    # 确定严重程度
    if risk_score < 30:
        severity = 'low'
    elif risk_score < 70:
        severity = 'medium'
    else:
        severity = 'high'

    return {
        'risk_score': float(risk_score),
        'confidence': float(confidence),
        'consistency': float(consistency),
        'severity': severity
    }


def fusion_analysis(
    face_predictions: List[Dict[str, float]],
    method: str = "top_vote",
    frequency_score: float = None,
    sync_score: float = None,
    llm_analysis: Dict = None
) -> Dict:
    """
    完整的融合分析

    Args:
        face_predictions: 人脸预测列表
        method: 融合方法
        frequency_score: Module B - FFT频域分析评分 (0-1, 可选)
        sync_score: Module B - 音视频同步评分 (0-1, 可选)
        llm_analysis: Module B - LLM分析结果 (可选)

    Returns:
        完整的分析结果
    """
    # 计算视频级风险
    video_result = calculate_video_risk(face_predictions, method)

    # 计算风险评分
    risk_result = calculate_risk_score(
        video_result['video_fake_prob'],
        video_result['num_faces'],
        face_predictions
    )

    # 合并结果
    result = {
        **video_result,
        **risk_result
    }

    # 添加统计信息
    if face_predictions:
        real_probs = [pred['real_prob'] for pred in face_predictions]
        fake_probs = [pred['fake_prob'] for pred in face_predictions]
        result['face_stats'] = {
            'mean_fake_prob': float(np.mean(fake_probs)),
            'std_fake_prob': float(np.std(fake_probs)),
            'min_fake_prob': float(np.min(fake_probs)),
            'max_fake_prob': float(np.max(fake_probs)),
            'median_fake_prob': float(np.median(fake_probs)),
            'mean_real_prob': float(np.mean(real_probs)),
            'std_real_prob': float(np.std(real_probs)),
            'min_real_prob': float(np.min(real_probs)),
            'max_real_prob': float(np.max(real_probs)),
            'median_real_prob': float(np.median(real_probs))
        }

    # Module B 集成字段（预留）
    result['module_b'] = {
        'frequency_score': frequency_score,
        'sync_score': sync_score,
        'llm_analysis': llm_analysis
    }

    # 如果有 Module B 数据，计算综合评分
    if frequency_score is not None or sync_score is not None:
        scores = [video_result['video_fake_prob']]
        if frequency_score is not None:
            scores.append(frequency_score)
        if sync_score is not None:
            scores.append(sync_score)

        result['combined_score'] = float(np.mean(scores))
        result['combined_label'] = 'fake' if result['combined_score'] >= config.THRESHOLD else 'real'

    return result


def compare_fusion_methods(
    face_predictions: List[Dict[str, float]]
) -> Dict[str, Dict]:
    """
    比较不同融合方法的结果

    Args:
        face_predictions: 人脸预测列表

    Returns:
        不同方法的结果字典
    """
    methods = ["average", "max", "top_vote", "weighted_average"]
    results = {}

    for method in methods:
        try:
            results[method] = fusion_analysis(face_predictions, method)
        except Exception as e:
            logging.error(f"融合方法 {method} 失败: {e}")
            results[method] = None

    return results
