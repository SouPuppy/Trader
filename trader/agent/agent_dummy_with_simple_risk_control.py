"""
带简单风控的 Dummy TradingAgent
演示如何使用 RiskManager 管道进行风险控制
"""
from typing import List, Dict, Optional
from trader.agent.TradingAgent import TradingAgent
from trader.backtest.engine import BacktestEngine
from trader.risk.OrderIntent import OrderIntent, OrderSide, PriceType
from trader.risk.RiskManager import RiskManagerPipeline, RiskContext
from trader.risk.control_leverage_limit import LeverageLimitRiskManager
from trader.logger import get_logger

logger = get_logger(__name__)


class DummyAgentWithSimpleRiskControl(TradingAgent):
    """
    带简单风控的 Dummy TradingAgent
    
    流程：
    1. Agent 计算 score/weight → 生成 OrderIntent(s)
    2. RiskManager Pipeline 验证和调整订单
    3. 将调整后的订单转换为 Engine 的 Action 执行
    """
    
    def __init__(self, name: str = "DummyAgentWithSimpleRiskControl",
                 max_position_weight: float = 0.1,
                 min_score_threshold: float = 0.0,
                 max_total_weight: float = 1.0,
                 max_leverage: float = 1.0,
                 trading_stocks: Optional[List[str]] = None):
        """
        初始化带风控的 Dummy Agent
        
        Args:
            name: TradingAgent 名称
            max_position_weight: 单个股票最大配置比例
            min_score_threshold: 最小 score 阈值
            max_total_weight: 所有股票总配置比例上限
            max_leverage: 最大杠杆率（传递给 LeverageLimitRiskManager）
            trading_stocks: 要交易的股票代码列表，如果为 None 则从持仓中推断
        """
        super().__init__(name, max_position_weight, min_score_threshold, max_total_weight)
        
        # 创建风险管理器管道
        self.risk_pipeline = RiskManagerPipeline([
            LeverageLimitRiskManager(
                name="LeverageLimit",
                max_leverage=max_leverage
            )
        ])
        
        # 设置要交易的股票列表
        self.trading_stocks = trading_stocks or []
    
    def score(self, stock_code: str, engine: BacktestEngine) -> float:
        """
        计算股票的 score（看好程度）
        
        使用与 DummyAgent 相同的逻辑
        
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
            
            # 简单的 score 计算
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
        
        改进版本：使用更激进的映射，让 score 更容易达到 max_position_weight
        
        Args:
            stock_code: 股票代码
            score: 该股票的 score 值
            engine: 回测引擎
            
        Returns:
            float: 资金配置比例 [0, max_position_weight]
        """
        # 如果 score 低于阈值，不配置
        if score < self.min_score_threshold:
            return 0.0
        
        # 改进的映射：使用平方根函数，让 score 更容易达到 max_position_weight
        # 例如：score=0.5 -> weight ≈ 0.7 * max_position_weight
        #      score=0.8 -> weight ≈ 0.9 * max_position_weight
        import math
        normalized_score = max(0.0, score)  # 只考虑正分
        # 使用平方根映射，让中等 score 也能获得较高权重
        weight = math.sqrt(normalized_score) * self.max_position_weight
        
        return min(weight, self.max_position_weight)
    
    def generate_order_intents(self, stock_weights: Dict[str, float], 
                              engine: BacktestEngine) -> List[OrderIntent]:
        """
        根据权重生成订单意图列表
        
        Args:
            stock_weights: {stock_code: weight} 权重字典
            engine: 回测引擎
            
        Returns:
            List[OrderIntent]: 订单意图列表
        """
        order_intents = []
        
        for stock_code, weight in stock_weights.items():
            # 获取当前持仓权重
            current_position = engine.account.get_position(stock_code)
            current_weight = 0.0
            
            if current_position:
                # 计算当前持仓权重
                market_prices = engine.get_market_prices([stock_code])
                if stock_code in market_prices:
                    equity = engine.account.equity(market_prices)
                    if equity > 0:
                        position_value = current_position["shares"] * market_prices[stock_code]
                        current_weight = position_value / equity
            
            # 计算目标权重变化
            target_weight_change = weight - current_weight
            
            # 如果目标权重变化很小，跳过（提高阈值到 0.5%）
            if abs(target_weight_change) < 0.005:  # 0.5% 阈值
                continue
            
            # 生成订单意图
            if target_weight_change > 0:
                # 买入（增加持仓）
                if weight > 0:  # 只有当目标权重 > 0 时才买入
                    order_intent = OrderIntent(
                        symbol=stock_code,
                        side=OrderSide.BUY,
                        timestamp=engine.current_date,
                        target_weight=weight,  # 目标总权重
                        price_type=PriceType.MKT,
                        agent_name=self.name,
                        confidence=abs(weight)  # 使用权重绝对值作为置信度
                    )
                    order_intents.append(order_intent)
            else:
                # 卖出（减少持仓或全部卖出）
                if current_position:
                    if weight <= 0:
                        # 如果目标权重 <= 0，全部卖出
                        sell_shares = current_position["shares"]
                    else:
                        # 如果目标权重 > 0 但小于当前权重，部分卖出
                        sell_weight = abs(target_weight_change)
                        # 转换为股数
                        market_prices = engine.get_market_prices([stock_code])
                        if stock_code in market_prices:
                            equity = engine.account.equity(market_prices)
                            target_value = equity * sell_weight
                            price = market_prices[stock_code]
                            sell_shares = int(target_value / price)
                        else:
                            continue
                    
                    if sell_shares > 0:
                        order_intent = OrderIntent(
                            symbol=stock_code,
                            side=OrderSide.SELL,
                            timestamp=engine.current_date,
                            qty=sell_shares,
                            price_type=PriceType.MKT,
                            agent_name=self.name,
                            confidence=abs(weight) if weight > 0 else 0.0,
                            reason_tags=["reduce_position"] if weight > 0 else ["close_position"]
                        )
                        order_intents.append(order_intent)
        
        return order_intents
    
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
        
        # 2. 调整订单
        adjusted_orders = []
        for order in validated_orders:
            adjusted_order = self.risk_pipeline.adjust(ctx, order)
            if adjusted_order.qty is not None and adjusted_order.qty > 0:
                adjusted_orders.append(adjusted_order)
            elif adjusted_order.target_weight is not None and adjusted_order.target_weight > 0:
                adjusted_orders.append(adjusted_order)
        
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
                            logger.info(
                                f"[{self.name}] 执行买入订单: {order.symbol}, "
                                f"目标权重 {order.target_weight:.2%}, "
                                f"买入金额 {buy_amount:.2f}"
                            )
                    elif order.qty is not None and order.qty > 0:
                        # 按股数买入
                        engine.buy(order.symbol, shares=order.qty)
                        logger.info(
                            f"[{self.name}] 执行买入订单: {order.symbol}, "
                            f"股数 {order.qty}"
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
    
    def set_trading_stocks(self, stock_codes: List[str]):
        """
        设置要交易的股票代码列表
        
        Args:
            stock_codes: 股票代码列表
        """
        self.trading_stocks = stock_codes
    
    def on_date(self, engine: BacktestEngine, date: str):
        """
        每个交易日的回调函数
        
        流程：
        1. 计算所有股票的 score 和 weight
        2. 生成订单意图
        3. 应用风险控制
        4. 执行订单
        
        Args:
            engine: 回测引擎
            date: 当前日期
        """
        # 确定要交易的股票列表
        # 优先使用设置的 trading_stocks，否则从持仓中推断
        stock_codes_to_trade = self.trading_stocks.copy() if self.trading_stocks else []
        
        # 如果没有设置，从持仓中获取
        if not stock_codes_to_trade:
            stock_codes_to_trade = list(engine.account.positions.keys())
        
        # 如果还是没有，跳过（无法确定要交易的股票）
        if not stock_codes_to_trade:
            return
        
        try:
            # 1. 计算所有股票的 score 和 weight
            stock_weights = {}
            for stock_code in stock_codes_to_trade:
                try:
                    score = self.score(stock_code, engine)
                    weight = self.weight(stock_code, score, engine)
                    # 即使 weight 为 0，也要记录，用于卖出逻辑
                    stock_weights[stock_code] = weight
                except Exception as e:
                    logger.warning(f"计算 {stock_code} 的 score/weight 时出错: {e}")
                    continue
            
            # 2. 确保所有持仓的股票都在 stock_weights 中（用于卖出逻辑）
            # 如果持仓的股票不在 stock_weights 中（比如 score 很低），设置 weight 为 0
            for stock_code in list(engine.account.positions.keys()):
                if stock_code not in stock_weights:
                    stock_weights[stock_code] = 0.0
            
            # 3. 归一化权重（确保总权重不超过 max_total_weight）
            # 只归一化 weight > 0 的股票
            positive_weights = {k: v for k, v in stock_weights.items() if v > 0}
            if positive_weights:
                normalized_positive = self.normalize_weights(positive_weights)
                # 更新 stock_weights
                for k, v in normalized_positive.items():
                    stock_weights[k] = v
            
            # 4. 生成订单意图（买入、卖出和调整）
            # generate_order_intents 会处理所有情况：买入、卖出、调整
            order_intents = self.generate_order_intents(stock_weights, engine)
            
            if not order_intents:
                return
            
            # 5. 应用风险控制
            approved_orders = self.apply_risk_control(order_intents, engine)
            
            if not approved_orders:
                return
            
            # 6. 执行订单
            self.execute_orders(approved_orders, engine)
            
            # 5. 记录交易后状态（如果有成交）
            # 注意：这里简化处理，实际应该从 engine 获取成交记录
            # fills = []  # 从 engine 获取成交记录
            # market_prices = engine.get_market_prices(stock_codes_to_trade)
            # ctx = RiskContext(
            #     account=engine.account,
            #     market=engine.market,
            #     current_date=engine.current_date,
            #     market_prices=market_prices
            # )
            # self.risk_pipeline.post_trade(ctx, fills)
        
        except Exception as e:
            logger.error(f"on_date 回调执行时出错: {e}", exc_info=True)

