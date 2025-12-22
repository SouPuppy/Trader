"""
Dummy TradingAgent 示例实现
一个简单的 TradingAgent 实现，用于演示和测试
支持定投策略（DCA）
"""
from typing import Dict, List, Optional
from datetime import datetime
from trader.agent.abstract_agent import AbstractAgent
from trader.backtest.engine import BacktestEngine
from trader.logger import get_logger

logger = get_logger(__name__)


class DummyAgent(AbstractAgent):
    """
    Dummy TradingAgent 示例
    使用简单的规则计算 score 和 weight
    支持定投策略（DCA - Dollar Cost Averaging）
    """
    
    def __init__(self, name: str = "DummyAgent",
                 max_position_weight: float = 0.1,
                 min_score_threshold: float = 0.0,
                 max_total_weight: float = 1.0,
                 dca_enabled: bool = True,
                 dca_amount: float = 1000.0,
                 dca_frequency: str = "monthly"):
        """
        初始化 Dummy TradingAgent
        
        Args:
            name: TradingAgent 名称
            max_position_weight: 单个股票最大配置比例
            min_score_threshold: 最小 score 阈值
            max_total_weight: 所有股票总配置比例上限
            dca_enabled: 是否启用定投策略
            dca_amount: 每次定投金额（元）
            dca_frequency: 定投频率，"monthly"（每月）或 "daily"（每日）
        """
        super().__init__(name, max_position_weight, min_score_threshold, max_total_weight)
        self.dca_enabled = dca_enabled
        self.dca_amount = dca_amount
        self.dca_frequency = dca_frequency
        self.last_dca_month: Optional[tuple] = None  # (year, month) 用于记录上次定投月份
        self.dca_stock_codes: List[str] = []  # 定投的股票代码列表
    
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
    
    def set_dca_stocks(self, stock_codes: List[str]):
        """
        设置定投的股票代码列表
        
        Args:
            stock_codes: 股票代码列表
        """
        self.dca_stock_codes = stock_codes
    
    def should_dca_today(self, date: str) -> bool:
        """
        判断今天是否应该执行定投
        
        Args:
            date: 当前日期
            
        Returns:
            bool: 是否应该执行定投
        """
        if not self.dca_enabled or not self.dca_stock_codes:
            return False
        
        # 解析日期
        try:
            date_obj = datetime.strptime(date, "%Y-%m-%d")
            current_month = (date_obj.year, date_obj.month)
        except ValueError:
            logger.warning(f"无法解析日期: {date}")
            return False
        
        # 根据定投频率决定是否执行定投
        if self.dca_frequency == "monthly":
            # 每月定投：如果是新的月份，执行定投
            if current_month != self.last_dca_month:
                self.last_dca_month = current_month
                return True
            return False
        elif self.dca_frequency == "daily":
            # 每日定投：每个交易日都定投
            return True
        else:
            logger.warning(f"未知的定投频率: {self.dca_frequency}")
            return False
    
    def on_date(self, engine: BacktestEngine, date: str):
        """
        每个交易日的回调函数
        
        实现定投策略：根据频率买入固定金额
        
        Args:
            engine: 回测引擎
            date: 当前日期
        """
        if not self.should_dca_today(date):
            return
        
        # 对每个股票执行定投
        for stock_code in self.dca_stock_codes:
            try:
                # 获取当前价格
                price = engine.get_current_price(stock_code)
                if price is None:
                    logger.warning(f"[{date}] 无法获取 {stock_code} 的价格，跳过定投")
                    continue
                
                # 执行定投买入
                engine.buy(stock_code, amount=self.dca_amount)
                logger.info(f"[{date}] 定投买入 {stock_code}: {self.dca_amount:.2f} 元 @ {price:.2f}")
            except Exception as e:
                logger.error(f"[{date}] 定投 {stock_code} 时出错: {e}", exc_info=True)

