"""
多资产交易代理
对多个股票分别使用独立的 agent 获取信号，然后进行权重归一化
"""
from typing import Dict, List, Optional
from trader.agent.TradingAgent import TradingAgent
from trader.agent.agent_logistic import LogisticAgent
from trader.agent.multiagent_weight_normalized import normalize_weights
from trader.backtest.engine import BacktestEngine
from trader.logger import get_logger

logger = get_logger(__name__)


class MultiAssetTradingAgent(TradingAgent):
    """
    多资产交易代理
    对每支股票使用独立的 logistic agent 获取信号，然后进行权重归一化
    """
    
    def __init__(
        self,
        stock_codes: List[str],
        name: str = "MultiAssetTradingAgent",
        max_position_weight: float = 0.2,
        min_score_threshold: float = 0.0,
        max_total_weight: float = 1.0,
        # LogisticAgent 参数
        feature_names: Optional[List[str]] = None,
        train_window_days: int = 252,
        prediction_horizon: int = 5,
        ret_threshold: float = 0.0,
        retrain_frequency: int = 20,
        train_test_split_ratio: float = 0.7
    ):
        """
        初始化多资产交易代理
        
        Args:
            stock_codes: 股票代码列表
            name: Agent 名称
            max_position_weight: 单个股票最大配置比例
            min_score_threshold: 最小 score 阈值
            max_total_weight: 总配置比例上限
            feature_names: LogisticAgent 使用的特征名称列表
            train_window_days: LogisticAgent 训练窗口大小
            prediction_horizon: LogisticAgent 预测周期
            ret_threshold: LogisticAgent 收益阈值
            retrain_frequency: LogisticAgent 重新训练频率
            train_test_split_ratio: LogisticAgent 训练/测试分割比例
        """
        super().__init__(
            name=name,
            max_position_weight=max_position_weight,
            min_score_threshold=min_score_threshold,
            max_total_weight=max_total_weight
        )
        
        self.stock_codes = stock_codes
        
        # 为每支股票创建独立的 LogisticAgent
        self.agents: Dict[str, LogisticAgent] = {}
        for stock_code in stock_codes:
            self.agents[stock_code] = LogisticAgent(
                name=f"{name}_{stock_code}",
                feature_names=feature_names,
                train_window_days=train_window_days,
                prediction_horizon=prediction_horizon,
                ret_threshold=ret_threshold,
                retrain_frequency=retrain_frequency,
                max_position_weight=max_position_weight,
                min_score_threshold=min_score_threshold,
                max_total_weight=max_total_weight,
                train_test_split_ratio=train_test_split_ratio
            )
        
        logger.info(
            f"创建 MultiAssetTradingAgent: {len(stock_codes)} 支股票, "
            f"每支股票使用独立的 LogisticAgent"
        )
    
    def score(self, stock_code: str, engine: BacktestEngine) -> float:
        """
        计算股票的 score（使用该股票对应的 LogisticAgent）
        
        Args:
            stock_code: 股票代码
            engine: 回测引擎
            
        Returns:
            float: 分数
        """
        if stock_code not in self.agents:
            logger.warning(f"股票 {stock_code} 不在代理列表中")
            return 0.0
        
        agent = self.agents[stock_code]
        return agent.score(stock_code, engine)
    
    def get_all_scores(self, engine: BacktestEngine) -> Dict[str, float]:
        """
        获取所有股票的 score
        
        Args:
            engine: 回测引擎
            
        Returns:
            Dict[str, float]: {stock_code: score} 分数字典
        """
        scores = {}
        for stock_code in self.stock_codes:
            scores[stock_code] = self.score(stock_code, engine)
        return scores
    
    def get_all_weights(self, engine: BacktestEngine) -> Dict[str, float]:
        """
        获取所有股票的权重（归一化后）
        
        Args:
            engine: 回测引擎
            
        Returns:
            Dict[str, float]: {stock_code: weight} 权重字典（已归一化）
        """
        # 获取所有股票的 score
        scores = self.get_all_scores(engine)
        
        # 计算每个股票的原始权重
        raw_weights = {}
        for stock_code, score in scores.items():
            agent = self.agents[stock_code]
            raw_weights[stock_code] = agent.weight(stock_code, score, engine)
        
        # 归一化权重
        normalized_weights = normalize_weights(raw_weights, self.max_total_weight)
        
        return normalized_weights
    
    def on_date(self, engine: BacktestEngine, date: str):
        """
        每个交易日的回调函数（更新所有 agent 的状态）
        
        Args:
            engine: 回测引擎
            date: 当前日期
        """
        for agent in self.agents.values():
            agent.on_date(engine, date)

