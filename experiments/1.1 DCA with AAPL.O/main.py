"""
DCA (Dollar Cost Averaging) backtest
Single-asset monthly investment strategy using DCAAgent
"""

import sys
from pathlib import Path

# Ensure project root is on PYTHONPATH
project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

from trader.agent.agent_dca import DCAAgent
from trader.backtest.account import Account
from trader.backtest.market import Market
from trader.backtest.engine import BacktestEngine
from trader.logger import get_logger, log_section, log_separator

logger = get_logger(__name__)


def run_dca_backtest(
    stock_code: str = "AAPL.O",
    initial_cash: float = 1_000_000.0,
    monthly_investment: float = 100_000.0,
    start_date: str | None = None,
    end_date: str | None = None,
):
    """Run a DCA backtest for a single asset."""

    for line in log_section("DCA Strategy Backtest"):
        logger.info(line)

    logger.info(f"Symbol: {stock_code}")
    logger.info(f"Initial cash: {initial_cash:,.2f}")
    logger.info(f"Monthly investment: {monthly_investment:,.2f}")

    market = Market(price_adjustment=0.01)
    account = Account(initial_cash=initial_cash)

    report_title = f"DCA_{stock_code}_monthly{int(monthly_investment)}"
    engine = BacktestEngine(account, market, report_title=report_title)

    available_dates = market.get_available_dates(stock_code)
    if not available_dates:
        logger.error(f"No market data for {stock_code}")
        return

    logger.info(f"Data range: {available_dates[0]} → {available_dates[-1]}")

    agent = DCAAgent(
        name="DCA",
        monthly_investment=monthly_investment,
        dca_frequency="monthly",
    )
    agent.set_dca_stocks([stock_code])

    def on_trading_day(engine: BacktestEngine, date: str):
        agent.on_date(engine, date)

    engine.on_date(on_trading_day)

    logger.info("Running backtest...")
    engine.run(stock_code, start_date=start_date, end_date=end_date)

    # 使用回测引擎实际运行的最后一个日期，或者最后一个可用日期
    # 如果 end_date 不在可用日期范围内，使用最后一个可用日期
    if end_date and end_date in available_dates:
        final_date = end_date
    else:
        # 使用回测引擎的最后一个交易日，或者可用日期的最后一个
        final_date = engine.current_date if engine.current_date else available_dates[-1]
    
    final_price = market.get_price(stock_code, final_date)
    if final_price is None:
        logger.error(f"Final price unavailable for date: {final_date}")
        logger.info(f"Available dates range: {available_dates[0]} → {available_dates[-1]}")
        return

    market_prices = {stock_code: final_price}
    equity = account.equity(market_prices)
    profit = account.get_total_profit(market_prices)
    return_pct = account.get_total_return(market_prices)
    invested = initial_cash - account.cash

    for line in log_section("Backtest Result"):
        logger.info(line)

    logger.info(f"DCA executions: {agent.investment_count}")
    logger.info(f"Capital invested: {invested:,.2f}")
    logger.info(f"Remaining cash: {account.cash:,.2f}")

    position = account.get_position(stock_code)
    if position:
        shares = position["shares"]
        avg_price = position["average_price"]
        value = shares * final_price
        pnl = (final_price - avg_price) * shares

        logger.info(f"Position: {shares} shares")
        logger.info(f"Avg cost: {avg_price:.2f}")
        logger.info(f"Market price: {final_price:.2f}")
        logger.info(f"Market value: {value:,.2f}")
        logger.info(f"Unrealized PnL: {pnl:+,.2f}")

    logger.info(f"Total equity: {equity:,.2f}")
    logger.info(f"Total PnL: {profit:+,.2f}")
    logger.info(f"Return: {return_pct:+.2f}%")
    logger.info(log_separator())

    logger.info(account.summary(market_prices))


if __name__ == "__main__":
    run_dca_backtest(
        stock_code="AAPL.O",
        start_date="2023-01-01",
        end_date="2023-12-31",
    )
