"""
杠杆限制风险管理器
限制账户总杠杆率，防止过度杠杆
"""
from typing import Tuple, Optional
from trader.risk.RiskManager import RiskManager, RiskContext
from trader.risk.OrderIntent import OrderIntent, OrderSide
from trader.logger import get_logger

logger = get_logger(__name__)


class LeverageLimitRiskManager(RiskManager):
    """
    杠杆限制风险管理器
    
    限制账户总杠杆率（总持仓市值 / 账户权益）
    防止过度杠杆导致的风险
    """
    
    def __init__(self, name: str = "LeverageLimitRiskManager", 
                 max_leverage: float = 1.0):
        """
        初始化杠杆限制风险管理器
        
        Args:
            name: 风险管理器名称
            max_leverage: 最大杠杆率（默认 1.0，即不允许杠杆）
                - 1.0: 不允许杠杆（总持仓市值 <= 账户权益）
                - 2.0: 允许 2 倍杠杆（总持仓市值 <= 2 * 账户权益）
        """
        super().__init__(name)
        self.max_leverage = max_leverage
    
    def validate(self, ctx: RiskContext, order: OrderIntent) -> Tuple[bool, Optional[str]]:
        """
        验证订单是否会导致杠杆超限
        
        Args:
            ctx: 风险控制上下文
            order: 订单意图
            
        Returns:
            Tuple[bool, Optional[str]]: (是否通过, 拒绝原因)
        """
        # 只检查买入订单（卖出订单会降低杠杆）
        if order.side != OrderSide.BUY:
            return True, None
        
        # 计算当前杠杆率
        current_leverage = self._calculate_leverage(ctx)
        
        # 如果当前杠杆已经超限，拒绝所有买入订单
        if current_leverage >= self.max_leverage:
            return False, f"当前杠杆率 {current_leverage:.2f} 已达到上限 {self.max_leverage:.2f}"
        
        # 估算订单执行后的杠杆率
        estimated_leverage = self._estimate_leverage_after_order(ctx, order)
        
        if estimated_leverage > self.max_leverage:
            return False, (
                f"订单执行后杠杆率 {estimated_leverage:.2f} 将超过上限 {self.max_leverage:.2f} "
                f"(当前杠杆率: {current_leverage:.2f})"
            )
        
        return True, None
    
    def adjust(self, ctx: RiskContext, order: OrderIntent) -> OrderIntent:
        """
        调整订单，确保不超过杠杆限制
        
        如果订单会导致杠杆超限，削减订单规模
        
        Args:
            ctx: 风险控制上下文
            order: 订单意图
            
        Returns:
            OrderIntent: 调整后的订单意图
        """
        # 只调整买入订单
        if order.side != OrderSide.BUY:
            return order
        
        # 计算当前杠杆率
        current_leverage = self._calculate_leverage(ctx)
        
        # 如果当前杠杆已经超限，拒绝订单（返回一个空订单）
        if current_leverage >= self.max_leverage:
            logger.warning(
                f"[{self.name}] 当前杠杆率 {current_leverage:.2f} 已达到上限 {self.max_leverage:.2f}，"
                f"拒绝订单: {order}"
            )
            # 返回一个数量为 0 的订单（相当于拒绝）
            adjusted_order = OrderIntent(
                symbol=order.symbol,
                side=order.side,
                timestamp=order.timestamp,
                qty=0,
                price_type=order.price_type,
                agent_name=order.agent_name,
                prompt_version=order.prompt_version,
                confidence=order.confidence,
                reason_tags=order.reason_tags,
                limit_price=order.limit_price,
                metadata={**order.metadata, "adjusted_by": self.name, "original_order": order.to_dict()}
            )
            return adjusted_order
        
        # 估算订单执行后的杠杆率
        estimated_leverage = self._estimate_leverage_after_order(ctx, order)
        
        if estimated_leverage <= self.max_leverage:
            # 不需要调整
            return order
        
        # 需要削减订单规模
        # 计算允许的最大杠杆增量
        max_leverage_increment = self.max_leverage - current_leverage
        
        # 计算当前持仓市值
        current_position_value = sum(
            ctx.get_position_value(symbol) or 0.0
            for symbol in ctx.account.positions.keys()
        )
        
        # 计算账户权益
        equity = ctx.get_account_equity()
        
        # 计算允许的最大持仓市值增量
        max_position_value_increment = max_leverage_increment * equity
        
        # 获取订单目标持仓市值
        order_target_value = self._get_order_target_value(ctx, order)
        
        if order_target_value <= max_position_value_increment:
            # 订单规模在允许范围内，不需要调整
            return order
        
        # 需要削减订单规模
        scale_factor = max_position_value_increment / order_target_value
        
        logger.info(
            f"[{self.name}] 削减订单规模: {order.symbol}, "
            f"原目标市值 {order_target_value:.2f}, "
            f"允许市值 {max_position_value_increment:.2f}, "
            f"缩放因子 {scale_factor:.2%}"
        )
        
        # 调整订单
        adjusted_order = self._scale_order(order, scale_factor)
        adjusted_order.metadata["adjusted_by"] = self.name
        adjusted_order.metadata["scale_factor"] = scale_factor
        adjusted_order.metadata["original_order"] = order.to_dict()
        
        return adjusted_order
    
    def _calculate_leverage(self, ctx: RiskContext) -> float:
        """
        计算当前杠杆率
        
        Args:
            ctx: 风险控制上下文
            
        Returns:
            float: 杠杆率（总持仓市值 / 账户权益）
        """
        equity = ctx.get_account_equity()
        if equity == 0:
            return 0.0
        
        # 计算总持仓市值
        total_position_value = sum(
            ctx.get_position_value(symbol) or 0.0
            for symbol in ctx.account.positions.keys()
        )
        
        leverage = total_position_value / equity
        return leverage
    
    def _estimate_leverage_after_order(self, ctx: RiskContext, order: OrderIntent) -> float:
        """
        估算订单执行后的杠杆率
        
        Args:
            ctx: 风险控制上下文
            order: 订单意图
            
        Returns:
            float: 估算的杠杆率
        """
        equity = ctx.get_account_equity()
        if equity == 0:
            return 0.0
        
        # 当前持仓市值
        current_position_value = sum(
            ctx.get_position_value(symbol) or 0.0
            for symbol in ctx.account.positions.keys()
        )
        
        # 订单目标持仓市值
        order_target_value = self._get_order_target_value(ctx, order)
        
        # 估算执行后的持仓市值
        # 注意：这里简化处理，假设订单完全执行
        # 实际应该考虑订单是否会影响现有持仓
        if order.symbol in ctx.account.positions:
            # 如果已有持仓，需要计算增量
            current_symbol_value = ctx.get_position_value(order.symbol) or 0.0
            estimated_symbol_value = current_symbol_value + order_target_value
            estimated_position_value = current_position_value - current_symbol_value + estimated_symbol_value
        else:
            # 新持仓
            estimated_position_value = current_position_value + order_target_value
        
        # 估算执行后的账户权益（买入会减少现金）
        estimated_equity = equity - order_target_value
        
        if estimated_equity <= 0:
            # 现金不足，杠杆率无限大（会被其他检查拒绝）
            return float('inf')
        
        estimated_leverage = estimated_position_value / estimated_equity
        return estimated_leverage
    
    def _get_order_target_value(self, ctx: RiskContext, order: OrderIntent) -> float:
        """
        获取订单目标持仓市值
        
        Args:
            ctx: 风险控制上下文
            order: 订单意图
            
        Returns:
            float: 目标持仓市值
        """
        price = ctx.market_prices.get(order.symbol)
        if price is None:
            # 如果无法获取价格，使用账户权益的 10% 作为估算
            equity = ctx.get_account_equity()
            logger.warning(
                f"[{self.name}] 无法获取 {order.symbol} 的价格，使用账户权益的 10% 作为估算"
            )
            return equity * 0.1
        
        if order.qty is not None:
            # 按股数计算
            return order.qty * price
        elif order.target_weight is not None:
            # 按权重计算
            equity = ctx.get_account_equity()
            return equity * order.target_weight
        else:
            # 不应该到达这里（OrderIntent 会验证）
            return 0.0
    
    def _scale_order(self, order: OrderIntent, scale_factor: float) -> OrderIntent:
        """
        缩放订单规模
        
        Args:
            order: 订单意图
            scale_factor: 缩放因子（0-1）
            
        Returns:
            OrderIntent: 缩放后的订单意图
        """
        if order.qty is not None:
            # 按股数缩放
            scaled_qty = max(0, int(order.qty * scale_factor))
            return OrderIntent(
                symbol=order.symbol,
                side=order.side,
                timestamp=order.timestamp,
                qty=scaled_qty,
                price_type=order.price_type,
                agent_name=order.agent_name,
                prompt_version=order.prompt_version,
                confidence=order.confidence,
                reason_tags=order.reason_tags,
                limit_price=order.limit_price,
                metadata=order.metadata.copy()
            )
        elif order.target_weight is not None:
            # 按权重缩放
            scaled_weight = order.target_weight * scale_factor
            return OrderIntent(
                symbol=order.symbol,
                side=order.side,
                timestamp=order.timestamp,
                target_weight=scaled_weight,
                price_type=order.price_type,
                agent_name=order.agent_name,
                prompt_version=order.prompt_version,
                confidence=order.confidence,
                reason_tags=order.reason_tags,
                limit_price=order.limit_price,
                metadata=order.metadata.copy()
            )
        else:
            # 不应该到达这里
            return order

