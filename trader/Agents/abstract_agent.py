"""
抽象 Agent 实现
提供一些通用的基础实现，方便子类继承
"""
from typing import Dict, List, Optional
from trader.Agents.Agent import Agent
from trader.backtest.engine import BacktestEngine
from trader.logger import get_logger

logger = get_logger(__name__)


class AbstractAgent(Agent):
    """
    抽象 Agent 实现
    提供一些通用的基础实现和工具方法
    """
    
    def __init__(self, name: str, 
                 max_position_weight: float = 0.1,
                 min_score_threshold: float = 0.0,
                 max_total_weight: float = 1.0):
        """
        初始化抽象 Agent
        
        Args:
            name: Agent 名称
            max_position_weight: 单个股票最大配置比例（默认10%）
            min_score_threshold: 最小 score 阈值，低于此值的股票 weight 为 0
            max_total_weight: 所有股票总配置比例上限（默认100%）
        """
        super().__init__(name)
        self.max_position_weight = max_position_weight
        self.min_score_threshold = min_score_threshold
        self.max_total_weight = max_total_weight
    
    def weight(self, stock_code: str, score: float, engine: BacktestEngine) -> float:
        """
        默认的 weight 计算实现
        基于 score 和风险控制参数计算配置比例
        
        Args:
            stock_code: 股票代码
            score: 该股票的 score 值
            engine: 回测引擎
            
        Returns:
            float: 资金配置比例 [0, max_position_weight]
        """
        # 如果 score 低于阈值，不配置
        if score < self.min_score_threshold:
            return 0.0
        
        # 简单的线性映射：score 越高，weight 越大
        # 将 score 映射到 [0, max_position_weight] 范围
        normalized_score = max(0.0, score)  # 只考虑正分
        weight = min(normalized_score * self.max_position_weight, self.max_position_weight)
        
        return weight
    
    def normalize_weights(self, weights: Dict[str, float]) -> Dict[str, float]:
        """
        归一化权重，确保总权重不超过 max_total_weight
        
        Args:
            weights: {stock_code: weight} 原始权重字典
            
        Returns:
            Dict[str, float]: 归一化后的权重字典
        """
        total_weight = sum(weights.values())
        
        if total_weight == 0:
            return weights
        
        # 如果总权重超过上限，按比例缩放
        if total_weight > self.max_total_weight:
            scale_factor = self.max_total_weight / total_weight
            normalized = {k: v * scale_factor for k, v in weights.items()}
        else:
            normalized = weights.copy()
        
        return normalized
    
    def filter_by_score(self, stock_scores: Dict[str, float], 
                       top_n: Optional[int] = None) -> Dict[str, float]:
        """
        根据 score 筛选股票
        
        Args:
            stock_scores: {stock_code: score} 分数字典
            top_n: 只保留前 N 个最高分的股票，如果为 None 则保留所有
            
        Returns:
            Dict[str, float]: 筛选后的分数字典
        """
        # 过滤掉低于阈值的股票
        filtered = {
            code: score for code, score in stock_scores.items()
            if score >= self.min_score_threshold
        }
        
        # 如果指定了 top_n，只保留前 N 个
        if top_n is not None and top_n > 0:
            sorted_items = sorted(filtered.items(), key=lambda x: x[1], reverse=True)
            filtered = dict(sorted_items[:top_n])
        
        return filtered

