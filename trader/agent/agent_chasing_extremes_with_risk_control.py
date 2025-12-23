"""
带风险控制的 Chasing Extremes Agent
在 Chasing Extremes Agent 基础上添加杠杆限制风险控制
"""
from typing import Dict, List, Optional
from trader.agent.agent_chasing_extremes import ChasingExtremesAgent
from trader.backtest.engine import BacktestEngine
from trader.risk.OrderIntent import OrderIntent, OrderSide, PriceType
from trader.risk.RiskManager import RiskManagerPipeline, RiskContext
from trader.risk.control_leverage_limit import LeverageLimitRiskManager
from trader.logger import get_logger

logger = get_logger(__name__)


class ChasingExtremesAgentWithRiskControl(ChasingExtremesAgent):
    """
    带风险控制的 Chasing Extremes Agent
    
    继承自 ChasingExtremesAgent，添加杠杆限制风险控制
    策略逻辑与 ChasingExtremesAgent 完全一致，只是增加了风险控制
    """
    
    def __init__(
        self,
        name: str = "ChasingExtremesAgentWithRiskControl",
        extreme_threshold: float = 0.05,
        lookback_days: int = 1,
        max_position_weight: float = 1.0,
        min_score_threshold: float = 0.0,
        max_total_weight: float = 1.0,
        chase_up: bool = True,
        chase_down: bool = True,
        max_leverage: float = 0.8
    ):
        """
        初始化带风险控制的 Chasing Extremes Agent
        
        Args:
            name: Agent 名称
            extreme_threshold: 极端波动阈值（如 0.05 表示 5%）
            lookback_days: 回看天数，用于计算涨跌幅
            max_position_weight: 最大仓位（默认全仓）
            min_score_threshold: 最小 score 阈值
            max_total_weight: 总配置比例上限（默认全仓）
            chase_up: 是否追涨（价格大涨时买入）
            chase_down: 是否追跌（价格大跌时卖出）
            max_leverage: 最大杠杆率（传递给 LeverageLimitRiskManager）
        """
        super().__init__(
            name=name,
            extreme_threshold=extreme_threshold,
            lookback_days=lookback_days,
            max_position_weight=max_position_weight,
            min_score_threshold=min_score_threshold,
            max_total_weight=max_total_weight,
            chase_up=chase_up,
            chase_down=chase_down
        )
        
        # 创建风险管理器管道
        self.risk_pipeline = RiskManagerPipeline([
            LeverageLimitRiskManager(
                name="LeverageLimit",
                max_leverage=max_leverage
            )
        ])
    
    def apply_risk_control(self, order_intents: List[OrderIntent], 
                          engine: BacktestEngine) -> List[OrderIntent]:
        """
        应用风险控制管道
        
        Args:
            order_intents: 订单意图列表
            engine: 回测引擎
            
        Returns:
            List[OrderIntent]: 通过风控后的订单意图列表
        """
        if not order_intents:
            return []
        
        # 创建风险控制上下文
        market_prices = {}
        for order in order_intents:
            if order.symbol not in market_prices:
                price = engine.get_price(order.symbol)
                if price is not None:
                    market_prices[order.symbol] = price
        
        # 获取所有持仓的市场价格
        for symbol in engine.account.positions.keys():
            if symbol not in market_prices:
                price = engine.get_price(symbol)
                if price is not None:
                    market_prices[symbol] = price
        
        ctx = RiskContext(
            account=engine.account,
            market=engine.market,
            current_date=engine.current_date,
            market_prices=market_prices
        )
        
        # 1. 验证订单（过滤掉被拒绝的订单）
        validated_orders = []
        for order in order_intents:
            ok, reason = self.risk_pipeline.validate(ctx, order)
            if ok:
                validated_orders.append(order)
            else:
                logger.warning(f"[{self.name}] 订单被拒绝: {order}, 原因: {reason}")
        
        # 2. 调整订单（削减规模以符合风险限制）
        adjusted_orders = []
        for order in validated_orders:
            adjusted_order = self.risk_pipeline.adjust(ctx, order)
            # 检查调整后的订单是否有效
            if adjusted_order.target_weight is not None and adjusted_order.target_weight > 0:
                adjusted_orders.append(adjusted_order)
                # 如果订单被削减，记录日志
                if order.target_weight is not None and adjusted_order.target_weight < order.target_weight:
                    logger.info(
                        f"[{self.name}] 订单被削减: {order.symbol}, "
                        f"原权重 {order.target_weight:.2%}, 调整后权重 {adjusted_order.target_weight:.2%}"
                    )
            elif adjusted_order.qty is not None and adjusted_order.qty > 0:
                adjusted_orders.append(adjusted_order)
                # 如果订单被削减，记录日志
                if order.qty is not None and adjusted_order.qty < order.qty:
                    logger.info(
                        f"[{self.name}] 订单被削减: {order.symbol}, "
                        f"原股数 {order.qty}, 调整后股数 {adjusted_order.qty}"
                    )
        
        # 3. 组合层面调整
        final_orders = self.risk_pipeline.pre_trade(ctx, adjusted_orders)
        
        return final_orders
    
    def execute_orders(self, order_intents: List[OrderIntent], engine: BacktestEngine):
        """
        执行订单意图（转换为 Engine 的 Action）
        
        Args:
            order_intents: 订单意图列表
            engine: 回测引擎
        """
        for order in order_intents:
            try:
                if order.side == OrderSide.BUY:
                    if order.target_weight is not None:
                        # 按目标权重买入
                        market_prices = engine.get_market_prices([order.symbol])
                        if order.symbol not in market_prices:
                            logger.warning(f"无法获取 {order.symbol} 的价格，跳过订单")
                            continue
                        
                        equity = engine.account.equity(market_prices)
                        target_value = equity * order.target_weight
                        
                        # 获取当前持仓市值
                        current_position = engine.account.get_position(order.symbol)
                        current_value = 0.0
                        if current_position and order.symbol in market_prices:
                            current_value = current_position["shares"] * market_prices[order.symbol]
                        
                        # 计算需要买入的金额
                        buy_amount = target_value - current_value
                        
                        if buy_amount > 0:
                            engine.buy(order.symbol, amount=buy_amount)
                            logger.debug(
                                f"[{self.name}] 执行买入订单: {order.symbol}, "
                                f"目标权重 {order.target_weight:.2%}, "
                                f"买入金额 {buy_amount:.2f}"
                            )
                    elif order.qty is not None and order.qty > 0:
                        # 按股数买入（备用逻辑）
                        engine.buy(order.symbol, shares=order.qty)
                        logger.debug(
                            f"[{self.name}] 执行买入订单: {order.symbol}, "
                            f"股数 {order.qty}"
                        )
                
                elif order.side == OrderSide.SELL:
                    if order.qty is not None and order.qty > 0:
                        engine.sell(order.symbol, shares=order.qty)
                        logger.debug(
                            f"[{self.name}] 执行卖出订单: {order.symbol}, "
                            f"股数 {order.qty}"
                        )
            
            except Exception as e:
                logger.error(f"执行订单时出错: {order}, 错误: {e}", exc_info=True)

