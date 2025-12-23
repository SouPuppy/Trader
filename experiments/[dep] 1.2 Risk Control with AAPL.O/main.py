"""
带风控的 DCA 策略回测示例
使用 DCAAgentWithRiskControl 实现带风险控制的定投策略
投资策略与 DCA 完全一致（每月固定金额买入），只是增加了风险控制
"""
import sys
from pathlib import Path

# 添加项目根目录到 Python 路径
project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from trader.agent.agent_dca_with_risk_control import DCAAgentWithRiskControl
from trader.backtest.account import Account
from trader.backtest.market import Market
from trader.backtest.engine import BacktestEngine
from trader.logger import get_logger, log_separator, log_section

logger = get_logger(__name__)


def risk_control_strategy(
    stock_code: str = "AAPL.O",
    initial_cash: float = 1000000.0,
    monthly_investment: float = 100000.0,
    max_leverage: float = 1.0,
    start_date: str = None,
    end_date: str = None
):
    """
    带风控的 DCA 策略回测（使用 DCAAgentWithRiskControl）
    
    投资策略与 DCA 完全一致（每月固定金额买入），只是增加了风险控制
    
    Args:
        stock_code: 股票代码
        initial_cash: 初始资金（元）
        monthly_investment: 每月定投金额（元）
        max_leverage: 最大杠杆率（1.0 = 不允许杠杆）
        start_date: 开始日期（格式: YYYY-MM-DD），如果为 None 则从最早可用日期开始
        end_date: 结束日期（格式: YYYY-MM-DD），如果为 None 则到最新日期
    """
    for line in log_section("带风控的 DCA 策略回测（使用 RiskManager）"):
        logger.info(line)
    logger.info(f"股票代码: {stock_code}")
    logger.info(f"初始资金: {initial_cash:,.2f} 元")
    logger.info(f"每月定投: {monthly_investment:,.2f} 元")
    logger.info(f"最大杠杆率: {max_leverage:.2f}")
    
    # 初始化市场、账户和回测引擎
    market = Market(price_adjustment=0.01)
    account = Account(initial_cash=initial_cash)
    
    # 生成报告标题（包含策略名称和参数，使用英文）
    report_title = (
        f"DCA_RiskControl_Strategy_{stock_code}_"
        f"monthly{int(monthly_investment)}_"
        f"maxLev{max_leverage:.2f}"
    )
    engine = BacktestEngine(account, market, report_title=report_title)
    
    # 获取可用日期（用于显示数据范围）
    available_dates = market.get_available_dates(stock_code)
    if not available_dates:
        logger.error(f"未找到股票 {stock_code} 的数据")
        return
    
    logger.info(f"数据范围: {available_dates[0]} 至 {available_dates[-1]}")
    
    # 创建带风控的 DCA Agent
    agent = DCAAgentWithRiskControl(
        name="DCA_RiskControl_Strategy",
        monthly_investment=monthly_investment,
        dca_frequency="monthly",
        max_leverage=max_leverage
    )
    
    # 设置定投的股票代码
    agent.set_dca_stocks([stock_code])
    
    # 注册交易日回调：调用 agent 的 on_date 方法执行策略
    def on_trading_day(eng: BacktestEngine, date: str):
        """每个交易日的回调"""
        agent.on_date(eng, date)
    
    engine.on_date(on_trading_day)
    
    # 运行回测
    logger.info("")
    logger.info("开始回测...")
    logger.info("")
    engine.run(stock_code, start_date=start_date, end_date=end_date)
    
    # 计算最终结果
    final_price = market.get_price(stock_code, end_date if end_date else available_dates[-1])
    if final_price is None:
        final_price = market.get_price(stock_code)  # 使用最新价格
    
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
    
    # 计算实际总投入（初始资金 - 剩余现金）
    total_invested = initial_cash - account.cash
    
    # 输出结果
    logger.info("")
    for line in log_section("回测结果"):
        logger.info(line)
    logger.info(f"定投次数: {agent.investment_count}")
    logger.info(f"初始资金: {initial_cash:,.2f} 元")
    logger.info(f"实际投入: {total_invested:,.2f} 元")
    logger.info(f"当前现金: {account.cash:,.2f} 元")
    logger.info(f"最终权益: {equity:,.2f} 元")
    logger.info(f"总盈亏: {profit:+,.2f} 元")
    logger.info(f"总收益率: {return_pct:+.2f}%")
    logger.info(f"交易次数: {len(account.trades)}")
    logger.info(f"当前杠杆率: {leverage:.2f} (上限: {max_leverage:.2f})")
    logger.info(f"持仓数量: {len(account.positions)}")
    
    # 输出持仓明细
    if account.positions:
        logger.info("")
        logger.info("持仓明细:")
        for symbol, position in account.positions.items():
            current_price = market_prices.get(symbol, position["average_price"])
            position_value = position["shares"] * current_price
            position_weight = position_value / equity if equity > 0 else 0.0
            profit_per_share = current_price - position["average_price"]
            total_profit = profit_per_share * position["shares"]
            
            logger.info(
                f"  {symbol}: {position['shares']} 股 @ {position['average_price']:.2f}, "
                f"现价 {current_price:.2f}, "
                f"市值 {position_value:,.2f} ({position_weight*100:.1f}%), "
                f"盈亏 {total_profit:+,.2f}"
            )
    
    logger.info(log_separator())
    
    # 输出详细账户摘要
    logger.info("")
    logger.info(account.summary(market_prices))


if __name__ == "__main__":
    # 执行带风控的 DCA 策略回测
    # 设置更严格的杠杆限制（0.6），让风险控制能够限制买入
    # 这样可以看到风险控制的效果：当杠杆率接近上限时，买入会被限制或削减
    risk_control_strategy(
        stock_code="AAPL.O",
        initial_cash=1000000.0,
        monthly_investment=100000.0,  # 每月定投金额，与 DCA 策略一致
        max_leverage=0.6,  # 设置更严格的杠杆限制，让风险控制起作用
        start_date="2023-01-03",  # 可以设置为 None 从最早日期开始
        end_date="2023-12-29"  # 可以设置为 None 到最新日期
    )

