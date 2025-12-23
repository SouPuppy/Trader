"""
多 Agent 权重归一化工具
用于将多个 agent 的权重进行归一化处理
"""
from typing import Dict, List
from trader.logger import get_logger

logger = get_logger(__name__)


def normalize_weights(
    weights: Dict[str, float],
    max_total_weight: float = 1.0
) -> Dict[str, float]:
    """
    归一化权重，确保总权重不超过 max_total_weight
    
    Args:
        weights: {stock_code: weight} 原始权重字典
        max_total_weight: 总权重上限（默认1.0，即100%）
        
    Returns:
        Dict[str, float]: 归一化后的权重字典
    """
    if not weights:
        return {}
    
    total_weight = sum(weights.values())
    
    if total_weight == 0:
        return weights
    
    # 如果总权重超过上限，按比例缩放
    if total_weight > max_total_weight:
        scale_factor = max_total_weight / total_weight
        normalized = {k: v * scale_factor for k, v in weights.items()}
        logger.debug(
            f"权重归一化: 总权重 {total_weight:.4f} > {max_total_weight:.4f}, "
            f"缩放因子: {scale_factor:.4f}"
        )
    else:
        normalized = weights.copy()
    
    return normalized


def combine_agent_weights(
    agent_weights_list: List[Dict[str, float]],
    max_total_weight: float = 1.0
) -> Dict[str, float]:
    """
    合并多个 agent 的权重，然后归一化
    
    Args:
        agent_weights_list: 多个 agent 的权重字典列表
        max_total_weight: 总权重上限（默认1.0，即100%）
        
    Returns:
        Dict[str, float]: 合并并归一化后的权重字典
    """
    if not agent_weights_list:
        return {}
    
    # 合并所有权重（简单相加）
    combined_weights = {}
    for weights in agent_weights_list:
        for stock_code, weight in weights.items():
            if stock_code not in combined_weights:
                combined_weights[stock_code] = 0.0
            combined_weights[stock_code] += weight
    
    # 归一化
    return normalize_weights(combined_weights, max_total_weight)

