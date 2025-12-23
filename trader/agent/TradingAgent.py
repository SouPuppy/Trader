"""
TradingAgent 交易代理基类
定义交易代理的接口，区分 score（研究问题）和 weight（工程+风控问题）

核心概念：
- score: 表达"看好程度/预期收益/排序依据"（研究问题）
         范围通常为 [-1, 1] 或 [0, 1]，用于排序和筛选
- weight: 表达"实际配置多少资金"（工程 + 风控问题）
          范围通常为 [0, 1]，表示资金配置比例，需要考虑风险控制

TradingAgent 提供了：
- score() 抽象方法：必须实现，计算股票的看好程度
- weight() 默认实现：基于 score 和风险控制参数，可以重写
- 工具方法：normalize_weights(), filter_by_score() 等
"""
from abc import ABC, abstractmethod
from typing import Dict, List, Optional
from trader.backtest.engine import BacktestEngine
from trader.logger import get_logger

logger = get_logger(__name__)


class TradingAgent(ABC):
    """
    交易代理基类
    
    核心概念：
    - score: 表达"看好程度/预期收益/排序依据"（研究问题）
           范围通常为 [-1, 1] 或 [0, 1]，用于排序和筛选
    - weight: 表达"实际配置多少资金"（工程 + 风控问题）
            范围通常为 [0, 1]，表示资金配置比例，需要考虑风险控制
    """
    
    def __init__(self, name: str,
                 max_position_weight: float = 0.1,
                 min_score_threshold: float = 0.0,
                 max_total_weight: float = 1.0):
        """
        初始化 TradingAgent
        
        Args:
            name: TradingAgent 名称
            max_position_weight: 单个股票最大配置比例（默认10%）
            min_score_threshold: 最小 score 阈值，低于此值的股票 weight 为 0
            max_total_weight: 所有股票总配置比例上限（默认100%）
        """
        self.name = name
        self.max_position_weight = max_position_weight
        self.min_score_threshold = min_score_threshold
        self.max_total_weight = max_total_weight
    
    @abstractmethod
    def score(self, stock_code: str, engine: BacktestEngine) -> float:
        """
        计算股票的看好程度/预期收益（研究问题）
        
        Args:
            stock_code: 股票代码
            engine: 回测引擎，用于获取数据和特征
            
        Returns:
            float: 分数，范围建议 [-1, 1] 或 [0, 1]
                  - 正数表示看好，负数表示看空
                  - 绝对值越大表示看好/看空程度越高
        """
        pass
    
    def weight(self, stock_code: str, score: float, engine: BacktestEngine) -> float:
        """
        计算实际资金配置比例（工程 + 风控问题）
        
        默认实现：基于 score 和风险控制参数计算配置比例
        可以重写此方法以添加自定义风控逻辑
        
        Args:
            stock_code: 股票代码
            score: 该股票的 score 值
            engine: 回测引擎，用于获取账户和市场信息
            
        Returns:
            float: 资金配置比例，范围 [0, max_position_weight]
                  - 0 表示不配置
                  - max_position_weight 表示达到单个股票上限
        """
        # 如果 score 低于阈值，不配置
        if score < self.min_score_threshold:
            return 0.0
        
        # 简单的线性映射：score 越高，weight 越大
        # 将 score 映射到 [0, max_position_weight] 范围
        normalized_score = max(0.0, score)  # 只考虑正分
        weight = min(normalized_score * self.max_position_weight, self.max_position_weight)
        
        return weight
    
    def get_scores(self, stock_codes: List[str], engine: BacktestEngine) -> Dict[str, float]:
        """
        批量计算多个股票的 score
        
        Args:
            stock_codes: 股票代码列表
            engine: 回测引擎
            
        Returns:
            Dict[str, float]: {stock_code: score} 分数字典
        """
        scores = {}
        for stock_code in stock_codes:
            try:
                scores[stock_code] = self.score(stock_code, engine)
            except Exception as e:
                logger.error(f"计算 {stock_code} 的 score 时出错: {e}", exc_info=True)
                scores[stock_code] = 0.0
        return scores
    
    def get_weights(self, stock_scores: Dict[str, float], 
                   engine: BacktestEngine) -> Dict[str, float]:
        """
        批量计算多个股票的 weight
        
        Args:
            stock_scores: {stock_code: score} 分数字典
            engine: 回测引擎
            
        Returns:
            Dict[str, float]: {stock_code: weight} 权重字典
        """
        weights = {}
        for stock_code, score in stock_scores.items():
            try:
                weights[stock_code] = self.weight(stock_code, score, engine)
            except Exception as e:
                logger.error(f"计算 {stock_code} 的 weight 时出错: {e}", exc_info=True)
                weights[stock_code] = 0.0
        return weights
    
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
    
    def on_date(self, engine: BacktestEngine, date: str):
        """
        每个交易日的回调函数（可选实现）
        
        Args:
            engine: 回测引擎
            date: 当前日期
        """
        pass
    
    def __repr__(self):
        return f"{self.__class__.__name__}(name='{self.name}')"
