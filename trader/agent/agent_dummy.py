"""
Dummy TradingAgent 示例实现
一个简单的 TradingAgent 实现，用于演示和测试
展示如何实现 score() 和 weight() 方法
"""
from trader.agent.TradingAgent import TradingAgent
from trader.backtest.engine import BacktestEngine
from trader.logger import get_logger

logger = get_logger(__name__)


class DummyAgent(TradingAgent):
    """
    Dummy TradingAgent 示例
    使用简单的规则计算 score 和 weight
    用于演示 TradingAgent 接口的基本用法
    """
    
    def __init__(self, name: str = "DummyAgent",
                 max_position_weight: float = 0.1,
                 min_score_threshold: float = 0.0,
                 max_total_weight: float = 1.0):
        """
        初始化 Dummy TradingAgent
        
        Args:
            name: TradingAgent 名称
            max_position_weight: 单个股票最大配置比例
            min_score_threshold: 最小 score 阈值
            max_total_weight: 所有股票总配置比例上限
        """
        super().__init__(name, max_position_weight, min_score_threshold, max_total_weight)
    
    def score(self, stock_code: str, engine: BacktestEngine) -> float:
        """
        计算股票的 score（看好程度）
        
        示例实现：基于简单的技术指标
        - 使用 ret_1d（1日收益率）作为基础
        - 使用 ret_20d（20日收益率）作为趋势
        - 组合计算 score
        
        Args:
            stock_code: 股票代码
            engine: 回测引擎
            
        Returns:
            float: score 值，范围 [-1, 1]
        """
        try:
            # 获取特征
            ret_1d = engine.get_feature("ret_1d", stock_code)
            ret_20d = engine.get_feature("ret_20d", stock_code)
            
            # 如果特征不可用，返回 0（中性）
            if ret_1d is None or ret_20d is None:
                return 0.0
            
            # 简单的 score 计算：
            # - 短期收益（ret_1d）权重 0.3
            # - 长期趋势（ret_20d）权重 0.7
            # - 使用 tanh 函数将收益率映射到 [-1, 1] 范围
            import math
            
            # 将收益率转换为 score（使用 tanh 进行平滑映射）
            score_1d = math.tanh(ret_1d * 10)  # 放大10倍后再 tanh
            score_20d = math.tanh(ret_20d * 5)  # 放大5倍后再 tanh
            
            # 加权组合
            score = 0.3 * score_1d + 0.7 * score_20d
            
            # 确保在 [-1, 1] 范围内
            score = max(-1.0, min(1.0, score))
            
            return float(score)
            
        except Exception as e:
            logger.error(f"计算 {stock_code} 的 score 时出错: {e}", exc_info=True)
            return 0.0
    
    def weight(self, stock_code: str, score: float, engine: BacktestEngine) -> float:
        """
        计算实际资金配置比例
        
        使用父类的默认实现，但可以在这里添加自定义逻辑
        
        Args:
            stock_code: 股票代码
            score: 该股票的 score 值
            engine: 回测引擎
            
        Returns:
            float: 资金配置比例 [0, max_position_weight]
        """
        # 使用父类的默认实现
        base_weight = super().weight(stock_code, score, engine)
        
        # 可以在这里添加额外的风控逻辑
        # 例如：检查持仓数量、账户风险等
        
        return base_weight

