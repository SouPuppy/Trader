"""
RAG-based Trading Agent
Uses RAG system to answer questions and make trading decisions
Leverages fixed RAG features: time window inference, data coverage checks, citation sanitization
"""
import re
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from pathlib import Path
import sys

project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from trader.agent.TradingAgent import TradingAgent
from trader.backtest.engine import BacktestEngine
from trader.rag.answer import rag_answer
from trader.rag.db.queries import ensure_tables
from trader.logger import get_logger

logger = get_logger(__name__)


class RAGTradingAgent(TradingAgent):
    """
    RAG-based Trading Agent
    
    Uses RAG system to analyze stocks and make trading decisions.
    Leverages the improved RAG system with:
    - Automatic time window inference from questions
    - Data coverage preflight checks
    - Time-aware retrieval for better temporal coverage
    - Citation sanitization for reliable answers
    """
    
    def __init__(self, name: str = "RAG Agent",
                 max_position_weight: float = 0.1,
                 min_score_threshold: float = 0.0,
                 max_total_weight: float = 1.0,
                 lookback_days: int = 30,
                 min_confidence: float = 0.3,
                 max_trades_per_year: int = 24,
                 run_interval_days: int = 10):
        """
        Initialize RAG Trading Agent
        
        Args:
            name: Agent name
            max_position_weight: Maximum position weight per stock
            min_score_threshold: Minimum score threshold for trading
            max_total_weight: Maximum total weight across all positions
            lookback_days: Number of days to look back for analysis (default 30)
            min_confidence: Minimum confidence threshold for trading decisions
            max_trades_per_year: Maximum number of trades allowed per year (default 24)
                                 Set to a large number (e.g., 100000) to disable limit for testing
            run_interval_days: Number of days between RAG agent runs (default 10)
        """
        super().__init__(name, max_position_weight, min_score_threshold, max_total_weight)
        self.lookback_days = lookback_days
        self.min_confidence = min_confidence
        self.max_trades_per_year = max_trades_per_year
        self.run_interval_days = run_interval_days
        
        # Ensure RAG database tables exist
        ensure_tables()
        
        # Cache for RAG results to avoid repeated queries on same day
        self._rag_cache: Dict[str, Dict] = {}
        
        # Track last run date for interval checking
        self._last_run_date: Optional[str] = None
    
    def score(self, stock_code: str, engine: BacktestEngine) -> float:
        """
        Calculate stock score using RAG system
        
        Uses multiple RAG queries to get comprehensive analysis:
        1. Market trend analysis (last N days)
        2. Risk assessment
        3. News impact analysis
        
        Args:
            stock_code: Stock code
            engine: Backtest engine
            
        Returns:
            float: Score in range [-1, 1]
                  - Positive: bullish signal (buy)
                  - Negative: bearish signal (sell)
                  - Near zero: neutral/hold
        """
        # Get current date from engine
        current_date = engine.current_date
        if not current_date:
            logger.warning("No current date in engine, using current time")
            current_date = datetime.now().strftime('%Y-%m-%d')
        
        # Check cache (same stock, same date)
        cache_key = f"{stock_code}:{current_date}"
        if cache_key in self._rag_cache:
            cached_result = self._rag_cache[cache_key]
            logger.debug(f"Using cached RAG result for {stock_code} on {current_date}")
            return cached_result['score']
        
        # Convert to ISO8601 format for RAG
        decision_time = f"{current_date}T00:00:00"
        
        # Query 1: Market trend analysis (uses automatic time window inference)
        trend_question = f"What is the market trend of {stock_code} in the last {self.lookback_days} days? Is it rising or falling?"
        
        # Query 2: Risk assessment
        risk_question = f"What are the current risks for {stock_code}? Should I be concerned?"
        
        scores = []
        weights = []
        
        try:
            # Query 1: Trend analysis
            trend_result = rag_answer(
                question=trend_question,
                stock_code=stock_code,
                decision_time=decision_time,
                frequency="1d"
            )
            
            trend_score = self._parse_trend_answer(trend_result.answer, trend_result.mode)
            scores.append(trend_score)
            weights.append(0.5)  # Trend analysis weight: 50%
            
            logger.debug(f"Trend analysis for {stock_code}: score={trend_score:.3f}, mode={trend_result.mode}")
            
            # Query 2: Risk assessment (only if trend is positive, to avoid over-trading)
            if trend_score > 0:
                risk_result = rag_answer(
                    question=risk_question,
                    stock_code=stock_code,
                    decision_time=decision_time,
                    frequency="1d"
                )
                
                risk_score = self._parse_risk_answer(risk_result.answer, risk_result.mode)
                scores.append(risk_score)
                weights.append(0.3)  # Risk assessment weight: 30%
                
                logger.debug(f"Risk assessment for {stock_code}: score={risk_score:.3f}, mode={risk_result.mode}")
            
            # Calculate weighted average score
            if scores and weights:
                total_weight = sum(weights)
                if total_weight > 0:
                    final_score = sum(s * w for s, w in zip(scores, weights)) / total_weight
                else:
                    final_score = 0.0
            else:
                final_score = 0.0
            
            # Adjust score based on RAG mode (degraded mode = lower confidence)
            if trend_result.mode == "degraded":
                final_score *= 0.5  # Reduce confidence if RAG is in degraded mode
            
            # Cache the result
            self._rag_cache[cache_key] = {
                'score': final_score,
                'trend_score': trend_score,
                'mode': trend_result.mode
            }
            
            logger.info(f"RAG score for {stock_code}: {final_score:.3f} (trend={trend_score:.3f}, mode={trend_result.mode})")
            
            return final_score
            
        except Exception as e:
            logger.error(f"RAG query failed for {stock_code}: {e}", exc_info=True)
            return 0.0  # Neutral on error
    
    def _parse_trend_answer(self, answer: str, mode: str) -> float:
        """
        Parse trend analysis answer to extract score
        
        Args:
            answer: RAG answer text
            mode: RAG mode (normal/degraded)
            
        Returns:
            float: Score in range [-1, 1]
        """
        answer_lower = answer.lower()
        
        # Strong positive signals
        strong_buy_signals = ['rising', 'upward', 'bullish', 'increasing', 'growing', 'positive trend', 'strong upward']
        # Moderate positive signals
        buy_signals = ['positive', 'good', 'strong', 'growth', 'improving', 'recovering']
        # Strong negative signals
        strong_sell_signals = ['falling', 'downward', 'bearish', 'declining', 'decreasing', 'negative trend', 'strong downward']
        # Moderate negative signals
        sell_signals = ['negative', 'bad', 'weak', 'decline', 'deteriorating', 'risky']
        # Neutral signals
        hold_signals = ['hold', 'neutral', 'stable', 'wait', 'uncertain', 'sideways', 'flat']
        
        # Check for strong signals first
        strong_buy_count = sum(1 for signal in strong_buy_signals if signal in answer_lower)
        strong_sell_count = sum(1 for signal in strong_sell_signals if signal in answer_lower)
        
        # Check for moderate signals
        buy_count = sum(1 for signal in buy_signals if signal in answer_lower)
        sell_count = sum(1 for signal in sell_signals if signal in answer_lower)
        hold_count = sum(1 for signal in hold_signals if signal in answer_lower)
        
        # Calculate score with stronger weight for strong signals
        if strong_buy_count > 0:
            score = 0.7 + min(0.3, strong_buy_count * 0.1)  # Range: [0.7, 1.0]
        elif strong_sell_count > 0:
            score = -0.7 - min(0.3, strong_sell_count * 0.1)  # Range: [-1.0, -0.7]
        elif buy_count > sell_count and buy_count > hold_count:
            score = 0.3 + min(0.4, buy_count * 0.1)  # Range: [0.3, 0.7]
        elif sell_count > buy_count and sell_count > hold_count:
            score = -0.3 - min(0.4, sell_count * 0.1)  # Range: [-0.7, -0.3]
        else:
            score = 0.0  # Neutral/hold
        
        # If degraded mode, reduce confidence
        if mode == "degraded":
            score *= 0.6
        
        return score
    
    def _parse_risk_answer(self, answer: str, mode: str) -> float:
        """
        Parse risk assessment answer to extract score
        
        Args:
            answer: RAG answer text
            mode: RAG mode (normal/degraded)
            
        Returns:
            float: Score adjustment in range [-1, 0] (negative = more risk = lower score)
        """
        answer_lower = answer.lower()
        
        # High risk signals (negative impact on score)
        high_risk_signals = ['high risk', 'very risky', 'dangerous', 'warning', 'concern', 'threat', 'volatile']
        # Moderate risk signals
        moderate_risk_signals = ['risk', 'uncertainty', 'caution', 'challenge', 'headwind']
        # Low risk signals (positive)
        low_risk_signals = ['low risk', 'stable', 'safe', 'secure', 'confident']
        
        high_risk_count = sum(1 for signal in high_risk_signals if signal in answer_lower)
        moderate_risk_count = sum(1 for signal in moderate_risk_signals if signal in answer_lower)
        low_risk_count = sum(1 for signal in low_risk_signals if signal in answer_lower)
        
        # Calculate risk score (negative = risk reduces overall score)
        if high_risk_count > 0:
            risk_score = -0.5 - min(0.3, high_risk_count * 0.1)  # Range: [-0.8, -0.5]
        elif moderate_risk_count > low_risk_count:
            risk_score = -0.2 - min(0.2, moderate_risk_count * 0.05)  # Range: [-0.4, -0.2]
        elif low_risk_count > 0:
            risk_score = 0.0  # Low risk doesn't penalize
        else:
            risk_score = -0.1  # Default slight negative for uncertainty
        
        return risk_score
    
    def weight(self, stock_code: str, score: float, engine: BacktestEngine) -> float:
        """
        Calculate position weight based on score and confidence
        
        Overrides parent class to add confidence-based weighting
        
        Args:
            stock_code: Stock code
            score: Score from RAG analysis
            engine: Backtest engine
            
        Returns:
            float: Position weight [0, max_position_weight]
        """
        # Use parent class weight calculation
        base_weight = super().weight(stock_code, score, engine)
        
        # Adjust based on score magnitude (confidence)
        # Higher absolute score = higher confidence = can use more weight
        confidence_multiplier = min(1.0, abs(score) / self.min_confidence) if self.min_confidence > 0 else 1.0
        
        adjusted_weight = base_weight * confidence_multiplier
        
        return adjusted_weight
    
    def on_date(self, engine: BacktestEngine, date: str):
        """
        Called each trading day to make trading decisions
        
        Args:
            engine: Backtest engine
            date: Current date
        """
        # Check if enough days have passed since last run
        if self._last_run_date is not None:
            try:
                last_dt = datetime.fromisoformat(self._last_run_date)
                current_dt = datetime.fromisoformat(f"{date}T00:00:00")
                days_since_last_run = (current_dt - last_dt).days
                
                if days_since_last_run < self.run_interval_days:
                    logger.debug(f"Skipping RAG agent on {date}: only {days_since_last_run} days since last run "
                               f"(need {self.run_interval_days} days)")
                    return
            except Exception as e:
                logger.warning(f"Failed to parse dates for interval check: {e}")
        
        # Update last run date (we will run RAG agent on this date)
        self._last_run_date = f"{date}T00:00:00"
        logger.info(f"Running RAG agent on {date} (every {self.run_interval_days} days)")
        
        # Check trading frequency limit
        decision_time = f"{date}T00:00:00"
        can_trade = self._can_trade(decision_time)
        if not can_trade:
            logger.debug(f"RAG agent ran on {date} but trading is limited (reached max trades limit)")
            # Still continue to evaluate positions (for logging/monitoring), but won't execute trades
        
        account = engine.account
        market = engine.market
        
        # Get current positions
        positions = account.get_all_positions()
        
        # Get market prices for all positions
        market_prices = {}
        for stock_code in positions.keys():
            price = market.get_price(stock_code, date)
            if price:
                market_prices[stock_code] = price
        
        # Step 1: Evaluate existing positions - decide whether to sell
        for stock_code, position in positions.items():
            if stock_code not in market_prices:
                continue
            
            # Check limit again before each trade
            if not can_trade:
                break
            
            shares = position["shares"]
            if shares <= 0:
                continue
            
            # Get score for this stock
            score = self.score(stock_code, engine)
            
            # Decision logic: sell if score is negative and below threshold
            if score < -self.min_confidence:
                if can_trade:
                    engine.sell(stock_code, shares)
                    logger.info(f"Sold {shares} shares of {stock_code} @ {market_prices[stock_code]:.2f} (score={score:.3f})")
                else:
                    logger.debug(f"Would sell {shares} shares of {stock_code} (score={score:.3f}) but trading is limited")
        
        # Step 2: Evaluate potential new positions
        # Get the stock code being traded in this backtest
        current_stock = getattr(engine, 'current_stock_code', None)
        if not current_stock:
            # Try to infer from engine's run() method
            # For single-stock backtest, use the stock being backtested
            if hasattr(engine, '_current_backtest_stock'):
                current_stock = engine._current_backtest_stock
            elif positions:
                # If we have positions, evaluate those stocks
                current_stock = list(positions.keys())[0]
            else:
                return  # No stock to evaluate
        
        # Check limit again before buying
        if not can_trade:
            return
        
        # Check if we have cash and don't already have a position
        if account.cash > 100 and current_stock not in positions:
            # Get score for current stock
            score = self.score(current_stock, engine)
            
            # Decision logic: buy if score is positive and above or equal to threshold
            if score >= self.min_confidence:
                price = market.get_price(current_stock, date)
                if price and price > 0:
                    # Calculate weight and shares to buy
                    weight = self.weight(current_stock, score, engine)
                    if weight > 0:
                        cash_to_use = account.cash * weight
                        shares = int(cash_to_use / price)
                        
                        if shares > 0:
                            if can_trade:
                                engine.buy(current_stock, shares=shares)
                                logger.info(f"Bought {shares} shares of {current_stock} @ {price:.2f} "
                                          f"(score={score:.3f}, weight={weight:.3f})")
                            else:
                                logger.debug(f"Would buy {shares} shares of {current_stock} @ {price:.2f} "
                                           f"(score={score:.3f}, weight={weight:.3f}) but trading is limited")
    
    def _get_trade_count_last_year(self, current_date: str) -> int:
        """
        Get number of trades in the last 365 days
        
        Args:
            current_date: Current date (ISO8601 format or YYYY-MM-DD)
            
        Returns:
            int: Number of trades in last 365 days
        """
        try:
            from trader.rag.db.queries import get_db_connection
            
            # Parse current date
            if 'T' in current_date:
                current_dt = datetime.fromisoformat(current_date.replace('Z', '+00:00'))
            else:
                current_dt = datetime.fromisoformat(f"{current_date}T00:00:00")
            
            # Calculate date 365 days ago
            date_1year_ago = (current_dt - timedelta(days=365)).isoformat()
            
            conn = get_db_connection()
            cursor = conn.cursor()
            cursor.execute("""
                SELECT COUNT(*) FROM trade_history
                WHERE trade_time >= ?
            """, (date_1year_ago,))
            count = cursor.fetchone()[0]
            conn.close()
            return count or 0
        except Exception as e:
            logger.error(f"Failed to get trade count: {e}", exc_info=True)
            return 0
    
    def _can_trade(self, current_date: str) -> bool:
        """
        Check if agent can make a trade (within frequency limits)
        
        Args:
            current_date: Current date (ISO8601 format or YYYY-MM-DD)
            
        Returns:
            bool: True if can trade, False otherwise
        """
        # If limit is very high, effectively disable limit
        if self.max_trades_per_year >= 100000:
            return True
        
        trade_count = self._get_trade_count_last_year(current_date)
        can_trade = trade_count < self.max_trades_per_year
        
        if not can_trade:
            logger.debug(f"Cannot trade on {current_date}: reached max trades limit "
                        f"({trade_count}/{self.max_trades_per_year} trades in last year)")
        
        return can_trade
    
    def clear_cache(self):
        """Clear RAG result cache"""
        self._rag_cache.clear()
        logger.debug("RAG cache cleared")


def run_rag_agent_backtest(
    stock_codes: List[str],
    start_date: str,
    end_date: str,
    initial_cash: float = 10000.0,
    lookback_days: int = 30,
    min_confidence: float = 0.3,
    max_trades_per_year: int = 24,
    run_interval_days: int = 10
):
    """
    Run backtest with RAG agent
    
    Args:
        stock_codes: List of stock codes to trade
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)
        initial_cash: Initial cash
        lookback_days: Number of days to look back for RAG analysis
        min_confidence: Minimum confidence threshold for trading
        max_trades_per_year: Maximum trades per year (default 24, set to 100000 to disable limit)
        run_interval_days: Number of days between RAG agent runs (default 10)
        
    Returns:
        BacktestReport: Generated report
    """
    from trader.backtest.engine import BacktestEngine
    from trader.backtest.account import Account
    from trader.backtest.market import Market
    from trader.backtest.report import BacktestReport
    from trader.config import PROJECT_ROOT
    
    # Create agent
    agent = RAGTradingAgent(
        name="RAG Agent",
        lookback_days=lookback_days,
        min_confidence=min_confidence,
        max_trades_per_year=max_trades_per_year,
        run_interval_days=run_interval_days
    )
    
    # Create report with timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    report_dir = PROJECT_ROOT / 'output' / 'rag_agent' / timestamp
    report_dir.mkdir(parents=True, exist_ok=True)
    report = BacktestReport(output_dir=report_dir, title=None)
    
    # Create account and market
    account = Account(initial_cash=initial_cash)
    market = Market()
    
    # Create engine with report enabled to record daily state
    engine = BacktestEngine(
        account=account,
        market=market,
        enable_report=True,  # Enable report to record daily state
        report_output_dir=report_dir,  # Use our report directory
        report_title=None,  # Don't auto-generate report file, we'll generate it manually
        train_test_split_ratio=0.0,  # No train/test split for RAG agent
        record_trade_history=True,  # Record trades to trade_history table
        clear_trade_history=True  # Clear trade_history at start (simulation mode)
    )
    # Replace engine's report with our report object to use the same instance
    engine.report = report
    
    # Register agent's on_date callback
    def on_trading_day(eng: BacktestEngine, date: str):
        # Set current stock code for agent
        eng.current_stock_code = stock_codes[0] if stock_codes else None
        eng._current_backtest_stock = stock_codes[0] if stock_codes else None
        # Call agent's on_date method
        agent.on_date(eng, date)
    
    engine.on_date(on_trading_day)
    
    # Run backtest for each stock
    for stock_code in stock_codes:
        logger.info(f"Running backtest for {stock_code}...")
        engine.run(stock_code, start_date=start_date, end_date=end_date)
    
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
