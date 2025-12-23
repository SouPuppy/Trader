"""
海龟策略（Turtle Trading Strategy）回测示例
使用 TurtleAgent 实现海龟策略
"""
import sys
from pathlib import Path

# 添加项目根目录到 Python 路径
project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from trader.agent.agent_turtle import TurtleAgent
from trader.backtest.account import Account
from trader.backtest.market import Market
from trader.backtest.engine import BacktestEngine
from trader.logger import get_logger, log_separator, log_section

logger = get_logger(__name__)


def turtle_strategy(
    stock_code: str = "AAPL.O",
    initial_cash: float = 1000000.0,  # 海龟策略需要更多资金
    entry_period: int = 20,
    exit_period: int = 10,
    atr_period: int = 20,
    risk_per_trade: float = 0.02,
    stop_loss_atr: float = 2.0,
    max_positions: int = 4,
    add_position_atr: float = 0.5,
    start_date: str = None,
    end_date: str = None
):
    """
    海龟策略回测（使用 TurtleAgent）
    
    Args:
        stock_code: 股票代码
        initial_cash: 初始资金
        entry_period: 突破周期
        exit_period: 退出周期
        atr_period: ATR计算周期
        risk_per_trade: 每次交易风险（账户资金的百分比）
        stop_loss_atr: 止损距离（ATR倍数）
        max_positions: 最大加仓次数
        add_position_atr: 加仓距离（ATR倍数）
        start_date: 开始日期
        end_date: 结束日期
    """
    for line in log_section("海龟策略回测（使用 Agent）"):
        logger.info(line)
    logger.info(f"股票代码: {stock_code}")
    logger.info(f"初始资金: {initial_cash:,.2f} 元")
    logger.info(f"突破周期: {entry_period} 天")
    logger.info(f"退出周期: {exit_period} 天")
    logger.info(f"ATR周期: {atr_period} 天")
    logger.info(f"风险比例: {risk_per_trade*100:.1f}%")
    logger.info(f"止损距离: {stop_loss_atr} ATR")
    logger.info(f"最大加仓次数: {max_positions}")
    
    # 初始化市场、账户和回测引擎
    market = Market(price_adjustment=0.01)
    account = Account(initial_cash=initial_cash)
    
    # 生成报告标题（包含策略名称和参数）
    report_title = f"Turtle_Strategy_{stock_code}_entry{entry_period}_exit{exit_period}_risk{risk_per_trade*100:.0f}%"
    engine = BacktestEngine(account, market, report_title=report_title)
    
    # 获取可用日期
    available_dates = market.get_available_dates(stock_code)
    if not available_dates:
        logger.error(f"未找到股票 {stock_code} 的数据")
        return
    
    logger.info(f"数据范围: {available_dates[0]} 至 {available_dates[-1]}")
    
    # 创建海龟策略 Agent
    agent = TurtleAgent(
        name="Turtle_Strategy",
        entry_period=entry_period,
        exit_period=exit_period,
        atr_period=atr_period,
        risk_per_trade=risk_per_trade,
        stop_loss_atr=stop_loss_atr,
        max_positions=max_positions,
        add_position_atr=add_position_atr
    )
    
    # 设置要交易的股票代码
    agent.set_trading_stocks([stock_code])
    
    # 注册交易日回调：调用 agent 的 on_date 方法执行策略
    def on_trading_day(eng: BacktestEngine, date: str):
        """每个交易日的回调"""
        agent.on_date(eng, date)
    
    engine.on_date(on_trading_day)
    
    # 运行回测
    logger.info("")
    logger.info("开始回测...")
    engine.run(stock_code, start_date=start_date, end_date=end_date)
    
    # 计算最终结果
    final_price = market.get_price(stock_code, end_date if end_date else available_dates[-1])
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
    # 执行海龟策略
    turtle_strategy(
        stock_code="AAPL.O",
        initial_cash=1000000.0,
        entry_period=20,
        exit_period=10,
        atr_period=20,
        risk_per_trade=0.02,
        stop_loss_atr=2.0,
        max_positions=4,
        add_position_atr=0.5,
        start_date=None,
        end_date=None
    )

