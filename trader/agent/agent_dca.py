"""
DCA (Dollar Cost Averaging) 定投策略 Agent
每月固定金额买入指定股票
"""
from typing import Dict, List, Optional
from datetime import datetime
from trader.agent.TradingAgent import TradingAgent
from trader.backtest.engine import BacktestEngine
from trader.logger import get_logger

logger = get_logger(__name__)


class DCAAgent(TradingAgent):
    """
    DCA 定投策略 Agent
    每月固定金额买入指定股票
    """
    
    def __init__(self, name: str = "DCAAgent",
                 monthly_investment: float = 1000.0,
                 dca_frequency: str = "monthly"):
        """
        初始化 DCA Agent
        
        Args:
            name: Agent 名称
            monthly_investment: 每月定投金额（元）
            dca_frequency: 定投频率，"monthly"（每月）或 "daily"（每日）
        """
        super().__init__(name, max_position_weight=1.0, max_total_weight=1.0)
        self.monthly_investment = monthly_investment
        self.dca_frequency = dca_frequency
        self.last_dca_month: Optional[tuple] = None  # (year, month) 用于记录上次定投月份
        self.dca_stock_codes: List[str] = []  # 定投的股票代码列表
        self.investment_count = 0  # 定投次数
    
    def score(self, stock_code: str, engine: BacktestEngine) -> float:
        """
        计算股票的 score（定投策略的看好程度）
        
        对于定投列表中的股票，返回 1.0（表示看好），其他股票返回 0.0
        
        Args:
            stock_code: 股票代码
            engine: 回测引擎
            
        Returns:
            float: 1.0 表示定投该股票，0.0 表示不定投
        """
        # 如果股票在定投列表中，返回 1.0（看好），否则返回 0.0
        if stock_code in self.dca_stock_codes:
            return 1.0
        return 0.0
    
    def weight(self, stock_code: str, score: float, engine: BacktestEngine) -> float:
        """
        计算实际资金配置比例（基于定投金额和账户权益）
        
        Args:
            stock_code: 股票代码
            score: 该股票的 score 值
            engine: 回测引擎
            
        Returns:
            float: 资金配置比例 [0, 1]
        """
        # 如果 score 为 0，不配置
        if score <= 0:
            return 0.0
        
        # 获取账户权益
        account_equity = engine.account.equity(engine.get_market_prices(self.dca_stock_codes))
        
        # 如果账户权益为 0，无法计算权重
        if account_equity <= 0:
            return 0.0
        
        # 计算定投金额占账户权益的比例
        # 注意：这里使用账户权益而不是初始资金，因为定投是持续投入
        weight = self.monthly_investment / account_equity
        
        # 限制在 [0, max_position_weight] 范围内
        weight = min(weight, self.max_position_weight)
        
        return weight
    
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
        if not self.dca_stock_codes:
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
        
        实现定投策略：直接使用固定金额买入，确保每次定投金额一致
        
        Args:
            engine: 回测引擎
            date: 当前日期
        """
        if not self.should_dca_today(date):
            return
        
        if not self.dca_stock_codes:
            return
        
        try:
            # 计算每个股票的定投金额（如果多个股票，平均分配）
            num_stocks = len(self.dca_stock_codes)
            investment_per_stock = self.monthly_investment / num_stocks
            
            for stock_code in self.dca_stock_codes:
                try:
                    # 获取当前价格
                    price = engine.get_current_price(stock_code)
                    if price is None:
                        logger.warning(f"[{date}] 无法获取 {stock_code} 的价格，跳过定投")
                        continue
                    
                    # 检查可用现金是否足够
                    if engine.account.cash < investment_per_stock:
                        logger.warning(
                            f"[{date}] 现金不足，无法定投 {stock_code}: "
                            f"需要 {investment_per_stock:.2f} 元，当前现金 {engine.account.cash:.2f} 元"
                        )
                        continue
                    
                    # 执行定投买入（使用固定金额）
                    engine.buy(stock_code, amount=investment_per_stock)
                    self.investment_count += 1
                    logger.info(
                        f"[{date}] 定投买入 {stock_code}: {investment_per_stock:.2f} 元 @ {price:.2f}"
                    )
                except Exception as e:
                    logger.error(f"[{date}] 定投 {stock_code} 时出错: {e}", exc_info=True)
        except Exception as e:
            logger.error(f"[{date}] 执行定投策略时出错: {e}", exc_info=True)

