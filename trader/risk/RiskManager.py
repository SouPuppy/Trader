"""
风险管理器接口
提供统一的风险管理接口，支持拒绝、修正、生成附加动作等能力
"""
from abc import ABC, abstractmethod
from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass, field
from trader.risk.OrderIntent import OrderIntent
from trader.backtest.account import Account
from trader.backtest.market import Market


@dataclass
class RiskContext:
    """
    风险控制上下文
    包含账户信息、市场信息等，用于风险决策
    """
    account: Account
    market: Market
    current_date: str  # 当前交易日，格式: YYYY-MM-DD
    market_prices: Dict[str, float]  # 当前市场价格 {symbol: price}
    
    # 可选：额外的上下文信息
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def get_account_equity(self) -> float:
        """获取账户总权益"""
        return self.account.equity(self.market_prices)
    
    def get_position_value(self, symbol: str) -> Optional[float]:
        """获取持仓市值"""
        position = self.account.get_position(symbol)
        if position is None:
            return None
        price = self.market_prices.get(symbol)
        if price is None:
            return None
        return position["shares"] * price
    
    def get_position_weight(self, symbol: str) -> float:
        """获取持仓权重（相对于总权益）"""
        equity = self.get_account_equity()
        if equity == 0:
            return 0.0
        position_value = self.get_position_value(symbol)
        if position_value is None:
            return 0.0
        return position_value / equity


class RiskManager(ABC):
    """
    风险管理器抽象基类
    
    提供三种能力：
    1. validate: 拒绝订单（返回 False）
    2. adjust: 修正订单（返回调整后的订单）
    3. pre_trade: 组合层面统一缩放（返回调整后的订单列表）
    4. post_trade: 记录风控状态（冷却期、连续亏损计数等）
    """
    
    def __init__(self, name: str):
        """
        初始化风险管理器
        
        Args:
            name: 风险管理器名称
        """
        self.name = name
    
    @abstractmethod
    def validate(self, ctx: RiskContext, order: OrderIntent) -> Tuple[bool, Optional[str]]:
        """
        验证订单是否可以通过
        
        Args:
            ctx: 风险控制上下文
            order: 订单意图
            
        Returns:
            Tuple[bool, Optional[str]]: (是否通过, 拒绝原因)
                - (True, None): 通过
                - (False, "reason"): 拒绝，原因说明
        """
        pass
    
    def adjust(self, ctx: RiskContext, order: OrderIntent) -> OrderIntent:
        """
        调整订单（例如削减仓位、截断换手）
        
        默认实现：不调整，直接返回原订单
        子类可以重写此方法以实现调整逻辑
        
        Args:
            ctx: 风险控制上下文
            order: 订单意图
            
        Returns:
            OrderIntent: 调整后的订单意图
        """
        return order
    
    def pre_trade(self, ctx: RiskContext, orders: List[OrderIntent]) -> List[OrderIntent]:
        """
        交易前组合层面统一缩放
        
        例如：波动率目标、总敞口上限等
        
        默认实现：不调整，直接返回原订单列表
        子类可以重写此方法以实现组合层面的调整
        
        Args:
            ctx: 风险控制上下文
            orders: 订单意图列表
            
        Returns:
            List[OrderIntent]: 调整后的订单意图列表
        """
        return orders
    
    def post_trade(self, ctx: RiskContext, fills: List[Dict[str, Any]]):
        """
        交易后记录风控状态
        
        例如：冷却期、连续亏损计数等
        
        默认实现：不执行任何操作
        子类可以重写此方法以实现状态记录
        
        Args:
            ctx: 风险控制上下文
            fills: 成交记录列表，每个元素包含：
                - symbol: 股票代码
                - side: 买卖方向
                - qty: 成交数量
                - price: 成交价格
                - timestamp: 成交时间
                - 其他元数据
        """
        pass
    
    def __repr__(self):
        return f"{self.__class__.__name__}(name='{self.name}')"


class RiskManagerPipeline:
    """
    风险管理器管道
    按顺序应用多个风险管理器
    """
    
    def __init__(self, risk_managers: List[RiskManager]):
        """
        初始化管道
        
        Args:
            risk_managers: 风险管理器列表，按顺序应用
        """
        self.risk_managers = risk_managers
    
    def validate(self, ctx: RiskContext, order: OrderIntent) -> Tuple[bool, Optional[str]]:
        """
        验证订单（所有风险管理器都必须通过）
        
        Args:
            ctx: 风险控制上下文
            order: 订单意图
            
        Returns:
            Tuple[bool, Optional[str]]: (是否通过, 拒绝原因)
        """
        for risk_manager in self.risk_managers:
            ok, reason = risk_manager.validate(ctx, order)
            if not ok:
                return False, f"{risk_manager.name}: {reason}"
        return True, None
    
    def adjust(self, ctx: RiskContext, order: OrderIntent) -> OrderIntent:
        """
        调整订单（依次应用所有风险管理器的调整）
        
        Args:
            ctx: 风险控制上下文
            order: 订单意图
            
        Returns:
            OrderIntent: 调整后的订单意图
        """
        adjusted_order = order
        for risk_manager in self.risk_managers:
            adjusted_order = risk_manager.adjust(ctx, adjusted_order)
        return adjusted_order
    
    def pre_trade(self, ctx: RiskContext, orders: List[OrderIntent]) -> List[OrderIntent]:
        """
        交易前组合层面调整（依次应用所有风险管理器的调整）
        
        Args:
            ctx: 风险控制上下文
            orders: 订单意图列表
            
        Returns:
            List[OrderIntent]: 调整后的订单意图列表
        """
        adjusted_orders = orders
        for risk_manager in self.risk_managers:
            adjusted_orders = risk_manager.pre_trade(ctx, adjusted_orders)
        return adjusted_orders
    
    def post_trade(self, ctx: RiskContext, fills: List[Dict[str, Any]]):
        """
        交易后记录状态（依次调用所有风险管理器）
        
        Args:
            ctx: 风险控制上下文
            fills: 成交记录列表
        """
        for risk_manager in self.risk_managers:
            risk_manager.post_trade(ctx, fills)
