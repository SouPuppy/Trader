"""
Chasing Extremes Agent 回测示例
这是一个"疯狂的" agent，用于测试 risk control 是否有用

策略逻辑：
- 当价格出现极端波动（大涨或大跌）时，会全仓追逐这个趋势
- 追涨：当价格大幅上涨时，全仓买入
- 追跌：当价格大幅下跌时，全仓卖出
- 没有风险控制，会全仓或大仓位买入/卖出
"""
import sys
from pathlib import Path

# 添加项目根目录到 Python 路径
project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from trader.agent.agent_chasing_extremes import ChasingExtremesAgent
from trader.backtest.account import Account
from trader.backtest.market import Market
from trader.backtest.engine import BacktestEngine
from trader.logger import get_logger, log_separator, log_section

logger = get_logger(__name__)


def chasing_extremes_backtest(
    stock_code: str = "AAPL.O",
    initial_cash: float = 1_000_000.0,
    extreme_threshold: float = 0.05,  # 极端波动阈值（5%）
    lookback_days: int = 1,  # 回看天数
    max_position_weight: float = 1.0,  # 最大仓位（全仓）
    chase_up: bool = True,  # 是否追涨
    chase_down: bool = True,  # 是否追跌
    start_date: str = None,
    end_date: str = None
):
    """
    Chasing Extremes Agent 回测
    
    Args:
        stock_code: 股票代码
        initial_cash: 初始资金
        extreme_threshold: 极端波动阈值（如 0.05 表示 5%）
        lookback_days: 回看天数，用于计算涨跌幅
        max_position_weight: 最大仓位（默认全仓）
        chase_up: 是否追涨（价格大涨时买入）
        chase_down: 是否追跌（价格大跌时卖出）
        start_date: 开始日期
        end_date: 结束日期
    """
    for line in log_section("Chasing Extremes Strategy Backtest"):
        logger.info(line)
    logger.info(f"股票代码: {stock_code}")
    logger.info(f"初始资金: {initial_cash:,.2f} 元")
    logger.info(f"极端波动阈值: {extreme_threshold*100:.1f}%")
    logger.info(f"回看天数: {lookback_days} 天")
    logger.info(f"最大仓位: {max_position_weight*100:.0f}%")
    logger.info(f"追涨: {chase_up}")
    logger.info(f"追跌: {chase_down}")
    
    # 初始化市场、账户和回测引擎
    market = Market(price_adjustment=0.01)
    account = Account(initial_cash=initial_cash)
    
    # 生成报告标题
    report_title = (
        f"ChasingExtremes_{stock_code}_"
        f"threshold{extreme_threshold*100:.0f}pct_"
        f"lookback{lookback_days}_"
        f"maxPos{max_position_weight*100:.0f}"
    )
    engine = BacktestEngine(account, market, report_title=report_title)
    
    # 获取可用日期
    available_dates = market.get_available_dates(stock_code)
    if not available_dates:
        logger.error(f"未找到股票 {stock_code} 的数据")
        return
    
    logger.info(f"数据范围: {available_dates[0]} 至 {available_dates[-1]}")
    
    # 创建 Chasing Extremes Agent
    agent = ChasingExtremesAgent(
        name="ChasingExtremes",
        extreme_threshold=extreme_threshold,
        lookback_days=lookback_days,
        max_position_weight=max_position_weight,
        chase_up=chase_up,
        chase_down=chase_down
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
        
        # 降低交易阈值，让它更频繁地交易（稳定亏钱）
        # 即使变化很小，只要有信号就交易
        min_trade_amount = 1000  # 最小交易金额（降低到1000元）
        if abs(diff_value) < min_trade_amount and weight > 0:
            # 如果权重>0但金额变化小，强制买入至少最小金额
            if diff_value > 0:
                diff_value = min_trade_amount
            else:
                # 卖出至少最小金额对应的股数
                if current_price and current_price > 0:
                    diff_value = -min_trade_amount
        
        # 执行交易
        if diff_value > min_trade_amount:
            # 买入
            eng.buy(stock_code, amount=diff_value)
            logger.info(
                f"[{date}] 追涨买入 {stock_code}: {diff_value:,.2f} 元 @ {current_price:.2f}, "
                f"score={score:.3f}, weight={weight:.3f}"
            )
        elif diff_value < -min_trade_amount:
            # 卖出
            shares_to_sell = int(abs(diff_value) / current_price) if current_price else 0
            if shares_to_sell > 0 and position:
                shares_to_sell = min(shares_to_sell, position['shares'])
                if shares_to_sell > 0:
                    eng.sell(stock_code, shares=shares_to_sell)
                    logger.info(
                        f"[{date}] 追跌卖出 {stock_code}: {shares_to_sell} 股 @ {current_price:.2f}, "
                        f"score={score:.3f}, weight={weight:.3f}"
                    )
    
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
    
    # 输出结果
    logger.info("")
    for line in log_section("回测结果"):
        logger.info(line)
    logger.info(f"初始资金: {initial_cash:,.2f} 元")
    logger.info(f"最终权益: {equity:,.2f} 元")
    logger.info(f"总盈亏: {profit:+,.2f} 元")
    logger.info(f"总收益率: {return_pct:+.2f}%")
    logger.info(f"交易次数: {len(account.trades)}")
    logger.info(log_separator())
    
    # 输出详细账户摘要
    logger.info("")
    logger.info(account.summary(market_prices))


if __name__ == "__main__":
    # 执行 Chasing Extremes 策略回测（稳定亏钱的疯狂策略）
    chasing_extremes_backtest(
        stock_code="AAPL.O",
        initial_cash=1_000_000.0,
        extreme_threshold=0.01,  # 1% 极端波动阈值（降低阈值，让它更容易触发）
        lookback_days=1,  # 回看1天
        max_position_weight=1.0,  # 全仓
        chase_up=True,  # 追涨（买在高点）
        chase_down=True,  # 追跌（卖在低点）
        start_date="2023-01-01",
        end_date="2023-12-31"
    )

