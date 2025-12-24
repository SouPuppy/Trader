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
    stock_codes = ["TSLA.O"]  # Single asset: TSLA.O only
    start_date = "2023-01-01"
    end_date = "2023-12-31"  # Use 2023 data only
    initial_cash = 10000.0
    lookback_days = 30  # Look back 30 days for trend analysis
    min_confidence = 0.3  # Minimum confidence threshold for trading
    max_trades_per_year = 24  # Maximum trades per year (set to 100000 to disable limit for testing)
    run_interval_days = 10  # Run RAG agent every 10 days
    
    logger.info("=" * 80)
    logger.info("RAG Agent Backtest")
    logger.info("=" * 80)
    logger.info(f"Stock codes: {stock_codes}")
    logger.info(f"Date range: {start_date} to {end_date}")
    logger.info(f"Initial cash: ${initial_cash:,.2f}")
    logger.info(f"Lookback days: {lookback_days}")
    logger.info(f"Min confidence: {min_confidence}")
    logger.info(f"Max trades per year: {max_trades_per_year}")
    logger.info(f"Run interval: Every {run_interval_days} days")
    logger.info("Note: Uses improved RAG system with time window inference and data coverage checks")
    logger.info("=" * 80)
    
    # Run backtest
    report = run_rag_agent_backtest(
        stock_codes=stock_codes,
        start_date=start_date,
        end_date=end_date,
        initial_cash=initial_cash,
        lookback_days=lookback_days,
        min_confidence=min_confidence,
        max_trades_per_year=max_trades_per_year,
        run_interval_days=run_interval_days
    )
    
    logger.info("Backtest completed!")
    logger.info(f"Report saved to: {report.output_dir}")


if __name__ == "__main__":
    main()

