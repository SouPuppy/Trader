"""
带风险控制的 DCA (Dollar Cost Averaging) 定投策略 Agent
每月固定金额买入指定股票，但通过 RiskManager 进行风险控制
"""
from typing import Dict, List, Optional
from datetime import datetime
from trader.agent.TradingAgent import TradingAgent
from trader.agent.agent_dca import DCAAgent
from trader.backtest.engine import BacktestEngine
from trader.risk.OrderIntent import OrderIntent, OrderSide, PriceType
from trader.risk.RiskManager import RiskManagerPipeline, RiskContext
from trader.risk.control_leverage_limit import LeverageLimitRiskManager
from trader.logger import get_logger

logger = get_logger(__name__)


class DCAAgentWithRiskControl(DCAAgent):
    """
    带风险控制的 DCA 定投策略 Agent
    
    继承自 DCAAgent，添加风险控制功能
    投资策略与 DCAAgent 完全一致（每月固定金额买入），只是增加了风险控制
    """
    
    def __init__(self, name: str = "DCAAgentWithRiskControl",
                 monthly_investment: float = 1000.0,
                 dca_frequency: str = "monthly",
                 max_leverage: float = 1.0):
        """
        初始化带风险控制的 DCA Agent
        
        Args:
            name: Agent 名称
            monthly_investment: 每月定投金额（元）
            dca_frequency: 定投频率，"monthly"（每月）或 "daily"（每日）
            max_leverage: 最大杠杆率（传递给 LeverageLimitRiskManager）
        """
        super().__init__(name, monthly_investment, dca_frequency)
        
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
            if adjusted_order.qty is not None and adjusted_order.qty > 0:
                adjusted_orders.append(adjusted_order)
                # 如果订单被削减，记录日志
                if order.qty is not None and adjusted_order.qty < order.qty:
                    logger.info(
                        f"[{self.name}] 订单被削减: {order.symbol}, "
                        f"原股数 {order.qty}, 调整后股数 {adjusted_order.qty}"
                    )
            elif adjusted_order.target_weight is not None and adjusted_order.target_weight > 0:
                adjusted_orders.append(adjusted_order)
                # 如果订单被削减，记录日志
                if order.target_weight is not None and adjusted_order.target_weight < order.target_weight:
                    logger.info(
                        f"[{self.name}] 订单被削减: {order.symbol}, "
                        f"原权重 {order.target_weight:.2%}, 调整后权重 {adjusted_order.target_weight:.2%}"
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
                    if order.qty is not None and order.qty > 0:
                        # 按股数买入（DCA 策略使用股数）
                        engine.buy(order.symbol, shares=order.qty)
                        dca_amount = order.metadata.get("dca_amount", 0.0)
                        logger.info(
                            f"[{self.name}] 执行买入订单: {order.symbol}, "
                            f"股数 {order.qty}, "
                            f"定投金额 {dca_amount:.2f}"
                        )
                    elif order.target_weight is not None:
                        # 按目标权重买入（备用逻辑）
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
                            logger.info(
                                f"[{self.name}] 执行买入订单: {order.symbol}, "
                                f"目标权重 {order.target_weight:.2%}, "
                                f"买入金额 {buy_amount:.2f}"
                            )
                
                elif order.side == OrderSide.SELL:
                    if order.qty is not None and order.qty > 0:
                        engine.sell(order.symbol, shares=order.qty)
                        logger.info(
                            f"[{self.name}] 执行卖出订单: {order.symbol}, "
                            f"股数 {order.qty}"
                        )
            
            except Exception as e:
                logger.error(f"执行订单时出错: {order}, 错误: {e}", exc_info=True)
    
    def on_date(self, engine: BacktestEngine, date: str):
        """
        每个交易日的回调函数
        
        实现带风险控制的定投策略：
        1. 判断是否应该定投
        2. 生成订单意图
        3. 应用风险控制
        4. 执行订单
        
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
            
            # 生成订单意图列表
            order_intents = []
            
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
                    
                    # 计算买入股数（基于固定金额）
                    # 使用 qty 而不是 target_weight，因为 DCA 策略的核心是固定金额增量买入
                    shares_to_buy = int(investment_per_stock / price)
                    
                    if shares_to_buy > 0:
                        # 生成买入订单意图（使用股数，这样风险控制可以正确计算增量市值）
                        order_intent = OrderIntent(
                            symbol=stock_code,
                            side=OrderSide.BUY,
                            timestamp=date,
                            qty=shares_to_buy,  # 使用股数，表示增量买入
                            price_type=PriceType.MKT,
                            agent_name=self.name,
                            confidence=1.0,  # DCA 策略置信度固定为 1.0
                            metadata={"dca_amount": investment_per_stock}  # 保存原始定投金额
                        )
                        order_intents.append(order_intent)
                    else:
                        logger.warning(
                            f"[{date}] 定投金额 {investment_per_stock:.2f} 元无法买入至少1股 "
                            f"(价格: {price:.2f})，跳过定投"
                        )
                    
                except Exception as e:
                    logger.error(f"[{date}] 生成 {stock_code} 订单意图时出错: {e}", exc_info=True)
            
            if not order_intents:
                return
            
            # 应用风险控制
            approved_orders = self.apply_risk_control(order_intents, engine)
            
            # 过滤掉被削减为 0 的订单
            valid_orders = []
            for order in approved_orders:
                if order.qty is not None and order.qty > 0:
                    valid_orders.append(order)
                elif order.target_weight is not None and order.target_weight > 0:
                    valid_orders.append(order)
            
            if not valid_orders:
                # 所有订单都被风险控制拒绝了
                logger.info(
                    f"[{date}] 所有定投订单都被风险控制拒绝，跳过本次定投"
                )
                return
            
            # 执行订单
            self.execute_orders(valid_orders, engine)
            
            # 更新定投次数（只统计实际执行的订单）
            # 注意：即使订单被削减，也算作一次定投尝试
            self.investment_count += len(order_intents)  # 使用原始订单数量，表示定投尝试次数
            
        except Exception as e:
            logger.error(f"[{date}] 执行定投策略时出错: {e}", exc_info=True)

