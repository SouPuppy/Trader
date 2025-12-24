"""
RAG-based Trading Agent
Uses RAG system to answer questions and make trading decisions
Limits trading to max 10 trades per 12 months to prevent LLM from getting stuck
"""
import sqlite3
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from pathlib import Path
import sys

project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from trader.agent.TradingAgent import TradingAgent
from trader.backtest.engine import BacktestEngine
from trader.backtest.account import Account
from trader.backtest.report import BacktestReport
from trader.rag.answer import rag_answer
from trader.rag.db.queries import get_db_connection, ensure_tables
from trader.config import DB_PATH, PROJECT_ROOT
from trader.logger import get_logger

logger = get_logger(__name__)


class RAGTradingAgent(TradingAgent):
    """
    RAG-based Trading Agent
    
    Uses RAG system to answer questions about stocks and make trading decisions.
    Limits trading to max 10 trades per 12 months to prevent LLM from getting stuck.
    """
    
    def __init__(self, name: str = "RAG Agent",
                 max_position_weight: float = 0.1,
                 min_score_threshold: float = 0.0,
                 max_total_weight: float = 1.0,
                 max_trades_per_12months: int = 10):
        """
        Initialize RAG Trading Agent
        
        Args:
            name: Agent name
            max_position_weight: Maximum position weight per stock
            min_score_threshold: Minimum score threshold
            max_total_weight: Maximum total weight
            max_trades_per_12months: Maximum number of trades allowed per 12 months
        """
        super().__init__(name, max_position_weight, min_score_threshold, max_total_weight)
        self.max_trades_per_12months = max_trades_per_12months
        
        # Ensure RAG database tables exist
        ensure_tables()
        
        # Track trades for this agent instance
        self.trade_count = 0
        self.last_trade_date = None
    
    def _record_trade(self, stock_code: str, action: str, price: float, volume: float, trade_time: str):
        """
        Record a trade to trade_history table
        
        Args:
            stock_code: Stock code
            action: Trade action ('BUY' or 'SELL')
            price: Trade price
            volume: Trade volume (shares)
            trade_time: Trade time (ISO8601 format)
        """
        try:
            conn = get_db_connection()
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO trade_history (stock_code, trade_time, action, price, volume)
                VALUES (?, ?, ?, ?, ?)
            """, (stock_code, trade_time, action, price, volume))
            conn.commit()
            conn.close()
            logger.info(f"Recorded trade: {action} {volume} shares of {stock_code} @ {price:.2f} at {trade_time}")
        except Exception as e:
            logger.error(f"Failed to record trade: {e}", exc_info=True)
    
    def _get_trade_count_last_12months(self, current_date: str) -> int:
        """
        Get number of trades in the last 12 months
        
        Args:
            current_date: Current date (ISO8601 format)
            
        Returns:
            int: Number of trades in last 12 months
        """
        try:
            # Parse current date
            current_dt = datetime.fromisoformat(current_date.replace('Z', '+00:00'))
            # Calculate date 12 months ago
            date_12months_ago = (current_dt - timedelta(days=365)).isoformat()
            
            conn = get_db_connection()
            cursor = conn.cursor()
            cursor.execute("""
                SELECT COUNT(*) FROM trade_history
                WHERE trade_time >= ?
            """, (date_12months_ago,))
            count = cursor.fetchone()[0]
            conn.close()
            return count
        except Exception as e:
            logger.error(f"Failed to get trade count: {e}", exc_info=True)
            return 0
    
    def _can_trade(self, current_date: str) -> bool:
        """
        Check if agent can make a trade (within limits)
        
        Args:
            current_date: Current date (ISO8601 format)
            
        Returns:
            bool: True if can trade, False otherwise
        """
        trade_count = self._get_trade_count_last_12months(current_date)
        return trade_count < self.max_trades_per_12months
    
    def score(self, stock_code: str, engine: BacktestEngine) -> float:
        """
        Calculate stock score using RAG system
        
        Args:
            stock_code: Stock code
            engine: Backtest engine
            
        Returns:
            float: Score in range [-1, 1]
        """
        # Get current date from engine
        current_date = engine.current_date
        if not current_date:
            logger.warning("No current date in engine, using current time")
            current_date = datetime.now().strftime('%Y-%m-%d')
        
        # Convert to ISO8601 format for RAG
        decision_time = f"{current_date}T00:00:00"
        
        # Ask RAG system about the stock
        # Use a question that helps determine if we should buy or sell
        question = f"What is the current market state and trend for {stock_code}? Should I buy, sell, or hold?"
        
        try:
            # Call RAG system
            result = rag_answer(
                question=question,
                stock_code=stock_code,
                decision_time=decision_time,
                frequency="1d"
            )
            
            # Parse answer to get score
            # Simple heuristic: look for keywords in answer
            answer_lower = result.answer.lower()
            
            # Positive signals
            buy_signals = ['buy', 'bullish', 'rising', 'upward', 'positive', 'good', 'strong', 'growth']
            # Negative signals
            sell_signals = ['sell', 'bearish', 'falling', 'downward', 'negative', 'bad', 'weak', 'decline']
            # Neutral signals
            hold_signals = ['hold', 'neutral', 'stable', 'wait', 'uncertain']
            
            # Count signals
            buy_count = sum(1 for signal in buy_signals if signal in answer_lower)
            sell_count = sum(1 for signal in sell_signals if signal in answer_lower)
            hold_count = sum(1 for signal in hold_signals if signal in answer_lower)
            
            # Calculate score: buy signals increase score, sell signals decrease it
            if buy_count > sell_count and buy_count > hold_count:
                score = 0.5 + min(0.5, buy_count * 0.1)  # Range: [0.5, 1.0]
            elif sell_count > buy_count and sell_count > hold_count:
                score = -0.5 - min(0.5, sell_count * 0.1)  # Range: [-1.0, -0.5]
            else:
                score = 0.0  # Neutral/hold
            
            logger.info(f"RAG score for {stock_code}: {score:.3f} (buy={buy_count}, sell={sell_count}, hold={hold_count})")
            
            return score
            
        except Exception as e:
            logger.error(f"RAG query failed for {stock_code}: {e}", exc_info=True)
            return 0.0  # Neutral on error
    
    def on_date(self, engine: BacktestEngine, date: str):
        """
        Called each trading day
        
        Args:
            engine: Backtest engine
            date: Current date
        """
        # Check if we can trade
        decision_time = f"{date}T00:00:00"
        if not self._can_trade(decision_time):
            logger.debug(f"Cannot trade on {date}: reached max trades limit ({self.max_trades_per_12months} per 12 months)")
            return
        
        # Get current positions
        account = engine.account
        positions = account.get_all_positions()
        
        # Get market prices
        market = engine.market
        market_prices = {}
        for stock_code in positions.keys():
            price = market.get_price(stock_code, date)
            if price:
                market_prices[stock_code] = price
        
        # Evaluate each position - decide whether to sell
        for stock_code, position in positions.items():
            if stock_code not in market_prices:
                continue
            
            current_price = market_prices[stock_code]
            shares = position["shares"]
            
            # Get score for this stock
            score = self.score(stock_code, engine)
            
            # Decision logic: if score is very negative, consider selling
            if score < -0.3 and shares > 0:
                # Check if we can still trade
                if not self._can_trade(decision_time):
                    break
                
                # Use engine's sell method (which will record to trade_history)
                engine.sell(stock_code, shares)
                self.trade_count += 1
                logger.info(f"Sold {shares} shares of {stock_code} @ {current_price:.2f} (score={score:.3f})")
        
        # Evaluate potential new positions
        # Get the stock code being traded in this backtest
        current_stock = getattr(engine, 'current_stock_code', None)
        if not current_stock:
            # Try to infer from positions or use first available
            if positions:
                current_stock = list(positions.keys())[0]
            else:
                return  # No stock to evaluate
        
        # Check if we have cash and can buy
        if account.cash > 100:  # Minimum cash threshold
            # Check if we can still trade
            if not self._can_trade(decision_time):
                return
            
            # Get score for current stock
            score = self.score(current_stock, engine)
            
            # If we don't have a position and score is positive, buy
            if current_stock not in positions and score > 0.3:
                price = market.get_price(current_stock, date)
                if price and price > 0:
                    # Calculate how many shares to buy (use weight from parent class)
                    weight = self.weight(current_stock, score, engine)
                    if weight > 0:
                        cash_to_use = account.cash * weight
                        shares = int(cash_to_use / price)
                        
                        if shares > 0:
                            # Use engine's buy method (which will record to trade_history)
                            engine.buy(current_stock, shares=shares)
                            self.trade_count += 1
                            logger.info(f"Bought {shares} shares of {current_stock} @ {price:.2f} (score={score:.3f}, weight={weight:.3f})")


def run_rag_agent_backtest(
    stock_codes: List[str],
    start_date: str,
    end_date: str,
    initial_cash: float = 10000.0,
    max_trades_per_12months: int = 10
):
    """
    Run backtest with RAG agent
    
    Args:
        stock_codes: List of stock codes to trade
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)
        initial_cash: Initial cash
        max_trades_per_12months: Maximum trades per 12 months
        
    Returns:
        BacktestReport: Generated report
    """
    from trader.backtest.engine import BacktestEngine
    
    # Create agent
    agent = RAGTradingAgent(
        name="RAG Agent",
        max_trades_per_12months=max_trades_per_12months
    )
    
    # Create report with timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    report_dir = PROJECT_ROOT / 'output' / 'rag_agent' / timestamp
    report_dir.mkdir(parents=True, exist_ok=True)
    # Pass None as title to avoid creating subdirectory
    report = BacktestReport(output_dir=report_dir, title=None)
    
    # Create account and market
    from trader.backtest.account import Account
    from trader.backtest.market import Market
    
    account = Account(initial_cash=initial_cash)
    market = Market()
    
    # Create engine (without report, we'll use our own report)
    # clear_trade_history=True by default (simulation mode)
    engine = BacktestEngine(
        account=account,
        market=market,
        enable_report=False,  # We'll use our own report
        train_test_split_ratio=0.0,  # No train/test split for RAG agent
        record_trade_history=True,  # Record trades to trade_history table
        clear_trade_history=True  # Clear trade_history at start (simulation mode)
    )
    
    # Register agent's on_date callback
    def on_trading_day(engine: BacktestEngine, date: str):
        # Set current stock code for agent (use the stock being backtested)
        engine.current_stock_code = getattr(engine, '_current_backtest_stock', stock_codes[0] if stock_codes else None)
        # Call agent's on_date method
        agent.on_date(engine, date)
    
    engine.on_date(on_trading_day)
    
    # Get all trading dates (union of all stocks' trading dates)
    all_dates = set()
    for stock_code in stock_codes:
        available_dates = market.get_available_dates(stock_code)
        if available_dates:
            for date in available_dates:
                if start_date <= date <= end_date:
                    all_dates.add(date)
    
    trading_dates = sorted(list(all_dates))
    
    # Manually iterate through dates (since we need to handle multiple stocks)
    for date in trading_dates:
        engine.current_date = date
        engine.date_index = trading_dates.index(date)
        
        # Set current stock for agent evaluation (use first stock)
        engine.current_stock_code = stock_codes[0] if stock_codes else None
        engine._current_backtest_stock = stock_codes[0] if stock_codes else None
        
        # Call agent's on_date method
        agent.on_date(engine, date)
        
        # Execute any pending actions
        engine._execute_actions()
        
        # Record daily state
        market_prices = {}
        for sc in stock_codes:
            price = market.get_price(sc, date)
            if price:
                market_prices[sc] = price
        
        if market_prices:
            report.record_daily_state(date, account, market_prices)
    
    # Generate report
    report.generate_report(
        account=account,
        stock_code=stock_codes[0] if stock_codes else "MULTI",
        start_date=start_date,
        end_date=end_date,
        all_stock_codes=stock_codes,
        is_multi_asset=len(stock_codes) > 1
    )
    
    logger.info(f"Report generated at: {report_dir}")
    
    return report

