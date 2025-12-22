"""
测试 TradingAgent 定投策略回测
使用 DummyAgent 实现定投策略并生成走势报告
"""
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from trader.agent import DummyAgent
from trader.backtest.account import Account
from trader.backtest.market import Market
from trader.backtest.engine import BacktestEngine
from trader.logger import get_logger, log_separator, log_section

logger = get_logger(__name__)


def test_agent():
    """使用 DummyAgent 进行定投策略回测"""
    for line in log_section("TradingAgent 定投策略回测"):
        logger.info(line)
    
    # 配置参数
    stock_code = "AAPL.O"
    initial_cash = 10000.0
    monthly_investment = 1000.0
    start_date = None  # 从最早日期开始
    end_date = None    # 到最新日期
    
    logger.info(f"股票代码: {stock_code}")
    logger.info(f"初始资金: {initial_cash:,.2f} 元")
    logger.info(f"每月定投: {monthly_investment:,.2f} 元")
    
    # 初始化市场、账户和回测引擎
    market = Market(price_adjustment=0.01)
    account = Account(initial_cash=initial_cash)
    engine = BacktestEngine(account, market, enable_report=True)  # 启用报告生成
    
    # 获取可用日期
    available_dates = market.get_available_dates(stock_code)
    if not available_dates:
        logger.error(f"未找到股票 {stock_code} 的数据")
        return
    
    logger.info(f"数据范围: {available_dates[0]} 至 {available_dates[-1]}")
    
    # 创建 DummyAgent，启用定投策略
    agent = DummyAgent(
        name="DCA_Agent",
        dca_enabled=True,
        dca_amount=monthly_investment,
        dca_frequency="monthly"  # 每月定投
    )
    
    # 设置定投的股票代码
    agent.set_dca_stocks([stock_code])
    
    # 注册交易日回调：调用 agent 的 on_date 方法执行定投
    def on_trading_day(eng: BacktestEngine, date: str):
        """每个交易日的回调"""
        # 调用 agent 的 on_date 方法，执行定投逻辑
        agent.on_date(eng, date)
    
    engine.on_date(on_trading_day)
    
    # 运行回测
    logger.info("")
    logger.info("开始回测...")
    engine.run(stock_code, start_date=start_date, end_date=end_date)
    
    # 计算最终结果
    final_date = end_date if end_date else available_dates[-1]
    final_price = market.get_price(stock_code, final_date)
    if final_price is None:
        final_price = market.get_price(stock_code)  # 使用最新价格
    
    if final_price is None:
        logger.error("无法获取最终价格")
        return
    
    market_prices = {stock_code: final_price}
    equity = account.equity(market_prices)
    profit = account.get_total_profit(market_prices)
    return_pct = account.get_total_return(market_prices)
    
    # 计算实际总投入
    total_invested = initial_cash - account.cash
    
    # 输出结果
    logger.info("")
    for line in log_section("回测结果"):
        logger.info(line)
    
    position = account.get_position(stock_code)
    if position:
        position_value = position["shares"] * final_price
        position_profit = (final_price - position["average_price"]) * position["shares"]
        logger.info(f"持仓股数: {position['shares']} 股")
        logger.info(f"平均成本: {position['average_price']:.2f} 元")
        logger.info(f"当前价格: {final_price:.2f} 元")
        logger.info(f"持仓市值: {position_value:,.2f} 元")
        logger.info(f"持仓盈亏: {position_profit:+,.2f} 元")
    
    logger.info(f"初始资金: {initial_cash:,.2f} 元")
    logger.info(f"实际投入: {total_invested:,.2f} 元")
    logger.info(f"当前现金: {account.cash:,.2f} 元")
    logger.info(f"总权益: {equity:,.2f} 元")
    logger.info(f"总盈亏: {profit:+,.2f} 元")
    logger.info(f"总收益率: {return_pct:+.2f}%")
    logger.info(log_separator())
    
    # 输出详细账户摘要
    logger.info("")
    logger.info(account.summary(market_prices))
    
    logger.info("")
    for line in log_section("回测完成"):
        logger.info(line)


if __name__ == "__main__":
    test_agent()

