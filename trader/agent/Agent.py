"""
Agent 接口模块
定义交易代理的接口，区分 score（研究问题）和 weight（工程+风控问题）
"""
from abc import ABC, abstractmethod
from typing import Dict, List, Optional
from trader.backtest.engine import BacktestEngine
from trader.logger import get_logger

logger = get_logger(__name__)


class Agent(ABC):
    """
    交易代理抽象基类
    
    核心概念：
    - score: 表达"看好程度/预期收益/排序依据"（研究问题）
           范围通常为 [-1, 1] 或 [0, 1]，用于排序和筛选
    - weight: 表达"实际配置多少资金"（工程 + 风控问题）
            范围通常为 [0, 1]，表示资金配置比例，需要考虑风险控制
    """
    
    def __init__(self, name: str):
        """
        初始化 Agent
        
        Args:
            name: Agent 名称
        """
        self.name = name
    
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
    
    @abstractmethod
    def weight(self, stock_code: str, score: float, engine: BacktestEngine) -> float:
        """
        计算实际资金配置比例（工程 + 风控问题）
        
        Args:
            stock_code: 股票代码
            score: 该股票的 score 值
            engine: 回测引擎，用于获取账户和市场信息
            
        Returns:
            float: 资金配置比例，范围 [0, 1]
                  - 0 表示不配置
                  - 1 表示全仓配置（通常不建议）
                  - 需要考虑风险控制、仓位限制等因素
        """
        pass
    
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
