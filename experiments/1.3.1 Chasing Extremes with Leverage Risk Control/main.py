"""
Chasing Extremes Agent 回测示例（带杠杆风险控制）
在 Chasing Extremes Agent 基础上添加杠杆限制风险控制，用于测试 risk control 是否有效

策略逻辑：
- 当价格出现极端波动（大涨或大跌）时，会全仓追逐这个趋势
- 追涨：当价格大幅上涨时，全仓买入
- 追跌：当价格大幅下跌时，全仓卖出
- 通过杠杆限制风险控制，限制总持仓市值不超过账户权益的一定比例
"""
import sys
from pathlib import Path

# 添加项目根目录到 Python 路径
project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from trader.agent.agent_chasing_extremes_with_risk_control import ChasingExtremesAgentWithRiskControl
from trader.backtest.account import Account
from trader.backtest.market import Market
from trader.backtest.engine import BacktestEngine
from trader.risk.OrderIntent import OrderIntent, OrderSide, PriceType
from trader.logger import get_logger, log_separator, log_section

logger = get_logger(__name__)


def chasing_extremes_with_risk_control_backtest(
    stock_code: str = "AAPL.O",
    initial_cash: float = 1_000_000.0,
    extreme_threshold: float = 0.05,  # 极端波动阈值（5%）
    lookback_days: int = 1,  # 回看天数
    max_position_weight: float = 1.0,  # 最大仓位（全仓）
    chase_up: bool = True,  # 是否追涨
    chase_down: bool = True,  # 是否追跌
    max_leverage: float = 0.8,  # 最大杠杆率（0.8 = 总持仓市值不超过账户权益的80%）
    start_date: str = None,
    end_date: str = None
):
    """
    Chasing Extremes Agent 回测（带杠杆风险控制）
    
    Args:
        stock_code: 股票代码
        initial_cash: 初始资金
        extreme_threshold: 极端波动阈值（如 0.05 表示 5%）
        lookback_days: 回看天数，用于计算涨跌幅
        max_position_weight: 最大仓位（默认全仓）
        chase_up: 是否追涨（价格大涨时买入）
        chase_down: 是否追跌（价格大跌时卖出）
        max_leverage: 最大杠杆率（传递给 LeverageLimitRiskManager）
        start_date: 开始日期
        end_date: 结束日期
    """
    for line in log_section("Chasing Extremes Strategy Backtest (with Leverage Risk Control)"):
        logger.info(line)
    logger.info(f"股票代码: {stock_code}")
    logger.info(f"初始资金: {initial_cash:,.2f} 元")
    logger.info(f"极端波动阈值: {extreme_threshold*100:.1f}%")
    logger.info(f"回看天数: {lookback_days} 天")
    logger.info(f"最大仓位: {max_position_weight*100:.0f}%")
    logger.info(f"追涨: {chase_up}")
    logger.info(f"追跌: {chase_down}")
    logger.info(f"最大杠杆率: {max_leverage:.2f}")
    
    # 初始化市场、账户和回测引擎
    market = Market(price_adjustment=0.01)
    account = Account(initial_cash=initial_cash)
    
    # 生成报告标题
    report_title = (
        f"ChasingExtremes_RiskControl_{stock_code}_"
        f"threshold{extreme_threshold*100:.0f}pct_"
        f"lookback{lookback_days}_"
        f"maxPos{max_position_weight*100:.0f}_"
        f"maxLev{max_leverage:.2f}"
    )
    engine = BacktestEngine(account, market, report_title=report_title)
    
    # 获取可用日期
    available_dates = market.get_available_dates(stock_code)
    if not available_dates:
        logger.error(f"未找到股票 {stock_code} 的数据")
        return
    
    logger.info(f"数据范围: {available_dates[0]} 至 {available_dates[-1]}")
    
    # 创建带风险控制的 Chasing Extremes Agent
    agent = ChasingExtremesAgentWithRiskControl(
        name="ChasingExtremes_RiskControl",
        extreme_threshold=extreme_threshold,
        lookback_days=lookback_days,
        max_position_weight=max_position_weight,
        chase_up=chase_up,
        chase_down=chase_down,
        max_leverage=max_leverage
    )
    
    # 注册交易日回调：执行策略
    def on_trading_day(eng: BacktestEngine, date: str):
        """每个交易日的回调"""
        # 更新 agent 状态
        agent.on_date(eng, date)
        
        # 获取当前股票的 score
        score = agent.score(stock_code, eng)
        
        # 计算权重
        weight = agent.weight(stock_code, score, eng)
        
        # 如果权重为0，检查是否需要卖出
        if weight == 0.0:
            position = account.get_position(stock_code)
            if position and position['shares'] > 0:
                # 如果 score < 0（追跌信号），卖出所有持仓
                if score < 0:
                    eng.sell(stock_code, shares=position['shares'])
                    logger.debug(f"[{date}] 追跌卖出 {stock_code}: {position['shares']} 股")
            return
        
        # 获取账户权益
        market_prices = eng.get_market_prices([stock_code])
        account_equity = account.equity(market_prices)
        
        # 计算目标持仓金额
        target_value = account_equity * weight
        
        # 获取当前持仓
        position = account.get_position(stock_code)
        current_value = 0.0
        current_price = eng.get_current_price(stock_code)
        if position and current_price:
            current_value = position['shares'] * current_price
        
        # 计算需要调整的金额
        diff_value = target_value - current_value
        
        # 如果变化很小，不需要交易
        if abs(diff_value) < 100:
            return
        
        # 生成订单意图
        order_intents = []
        if diff_value > 0:
            # 买入
            order_intent = OrderIntent(
                symbol=stock_code,
                side=OrderSide.BUY,
                timestamp=date,
                target_weight=weight,  # 使用目标权重
                price_type=PriceType.MKT,
                agent_name=agent.name,
                confidence=abs(score),  # 使用 score 的绝对值作为置信度
                metadata={"score": score, "diff_value": diff_value}
            )
            order_intents.append(order_intent)
        else:
            # 卖出
            shares_to_sell = int(abs(diff_value) / current_price) if current_price else 0
            if shares_to_sell > 0 and position:
                # 不能卖出超过持仓数量
                shares_to_sell = min(shares_to_sell, position['shares'])
                if shares_to_sell > 0:
                    order_intent = OrderIntent(
                        symbol=stock_code,
                        side=OrderSide.SELL,
                        timestamp=date,
                        qty=shares_to_sell,
                        price_type=PriceType.MKT,
                        agent_name=agent.name,
                        confidence=1.0,
                        metadata={"score": score, "diff_value": diff_value}
                    )
                    order_intents.append(order_intent)
        
        # 应用风险控制（只对买入订单应用）
        if order_intents:
            approved_orders = agent.apply_risk_control(order_intents, eng)
            
            # 执行订单
            agent.execute_orders(approved_orders, eng)
    
    engine.on_date(on_trading_day)
    
    # 运行回测
    logger.info("")
    logger.info("开始回测...")
    engine.run(stock_code, start_date=start_date, end_date=end_date)
    
    # 计算最终结果
    final_date = end_date if end_date else available_dates[-1]
    final_price = market.get_price(stock_code, final_date)
    if final_price is None:
        final_price = market.get_price(stock_code)
    
    if final_price is None:
        logger.error("无法获取最终价格")
        return
    
    market_prices = {stock_code: final_price}
    equity = account.equity(market_prices)
    profit = account.get_total_profit(market_prices)
    return_pct = account.get_total_return(market_prices)
    
    # 计算杠杆率
    total_position_value = sum(
        position["shares"] * market_prices.get(symbol, position["average_price"])
        for symbol, position in account.positions.items()
    )
    leverage = total_position_value / equity if equity > 0 else 0.0
    
    # 输出结果
    logger.info("")
    for line in log_section("回测结果"):
        logger.info(line)
    logger.info(f"初始资金: {initial_cash:,.2f} 元")
    logger.info(f"最终权益: {equity:,.2f} 元")
    logger.info(f"总盈亏: {profit:+,.2f} 元")
    logger.info(f"总收益率: {return_pct:+.2f}%")
    logger.info(f"交易次数: {len(account.trades)}")
    logger.info(f"当前杠杆率: {leverage:.2f} (上限: {max_leverage:.2f})")
    logger.info(log_separator())
    
    # 输出详细账户摘要
    logger.info("")
    logger.info(account.summary(market_prices))


if __name__ == "__main__":
    # 执行 Chasing Extremes 策略回测（带杠杆风险控制）
    chasing_extremes_with_risk_control_backtest(
        stock_code="AAPL.O",
        initial_cash=1_000_000.0,
        extreme_threshold=0.05,  # 5% 极端波动阈值
        lookback_days=1,  # 回看1天
        max_position_weight=1.0,  # 全仓
        chase_up=True,  # 追涨
        chase_down=True,  # 追跌
        max_leverage=0.8,  # 最大杠杆率（0.8 = 总持仓市值不超过账户权益的80%）
        start_date="2023-01-01",
        end_date="2023-12-31"
    )

