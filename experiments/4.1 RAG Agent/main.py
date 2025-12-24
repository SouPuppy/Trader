"""
RAG Agent Backtest Experiment
"""
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from trader.agent.agent_rag import run_rag_agent_backtest
from trader.logger import get_logger

logger = get_logger(__name__)


def main():
    """Run RAG agent backtest"""
    
    # Configuration
    stock_codes = ["AAPL.O", "TSLA.O", "NVDA.O"]  # Example stocks
    start_date = "2023-01-01"
    end_date = "2024-12-31"
    initial_cash = 10000.0
    max_trades_per_12months = 10
    
    logger.info("=" * 80)
    logger.info("RAG Agent Backtest")
    logger.info("=" * 80)
    logger.info(f"Stock codes: {stock_codes}")
    logger.info(f"Date range: {start_date} to {end_date}")
    logger.info(f"Initial cash: ${initial_cash:,.2f}")
    logger.info(f"Max trades per 12 months: {max_trades_per_12months}")
    logger.info("Note: trade_history will be cleared at start (simulation mode)")
    logger.info("=" * 80)
    
    # Run backtest
    report = run_rag_agent_backtest(
        stock_codes=stock_codes,
        start_date=start_date,
        end_date=end_date,
        initial_cash=initial_cash,
        max_trades_per_12months=max_trades_per_12months
    )
    
    logger.info("Backtest completed!")
    logger.info(f"Report saved to: {report.output_dir}")


if __name__ == "__main__":
    main()

