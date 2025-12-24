"""
Retrieval Planner
Generate retrieval plan JSON, decide which doc_types, time windows, k values, constraints and query forms
"""
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import re
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from trader.rag.types import (
    RagRequest, RetrievalPlan, RetrievalNeed, DocType, TaskType
)
from trader.logger import get_logger

logger = get_logger(__name__)


def build_plan(request: RagRequest) -> RetrievalPlan:
    """
    Build retrieval plan
    
    Args:
        request: RAG request
        
    Returns:
        RetrievalPlan
    """
    # Parse decision time
    try:
        decision_time = datetime.fromisoformat(request.decision_time.replace('Z', '+00:00'))
    except (ValueError, AttributeError):
        decision_time = datetime.now()
    
    # Infer task type from question type
    task_type = _infer_task_type(request.question)
    
    # Set time window based on task type and question text
    time_window_days = _get_time_window(task_type, request.question)
    time_end = decision_time.isoformat()
    time_start = (decision_time - timedelta(days=time_window_days)).isoformat()
    
    # Set retrieval needs based on task type
    needs = _build_retrieval_needs(task_type, request)
    
    # Build constraints
    constraints = _build_constraints(task_type, request)
    
    # Build semantic queries (for news)
    semantic_queries = _build_semantic_queries(request.question, task_type)
    
    # Build constraint queries (for SQL filtering)
    constraint_queries = _build_constraint_queries(request, task_type)
    
    plan = RetrievalPlan(
        task_type=task_type,
        stock_code=request.stock_code,
        time_start=time_start,
        time_end=time_end,
        needs=needs,
        constraints=constraints,
        semantic_queries=semantic_queries,
        constraint_queries=constraint_queries
    )
    
    logger.info(f"Built retrieval plan: task_type={task_type}, stock_code={request.stock_code}, "
                f"time_window=[{time_start}, {time_end}]")
    
    return plan


def _infer_task_type(question: str) -> TaskType:
    """
    Infer task type from question with strict classification rules
    
    Priority order (most specific first):
    1. news_impact: Contains news/event/catalyst keywords
    2. trade_explain: Contains trading history keywords
    3. market_state: Contains trend/market state keywords
    4. risk_check: Contains risk keywords
    5. strategy_suggest: Contains strategy/suggestion keywords
    """
    question_lower = question.lower()
    
    # 1. NEWS_IMPACT: Highest priority - check for news/event/catalyst keywords first
    # These should NOT fall through to market_state
    news_impact_keywords = [
        'news', 'important news', 'impact', 'headline', 'event', 'catalyst',
        'announcement', 'press release', 'news event', 'news impact',
        'what news', 'recent news', 'news about', 'news regarding'
    ]
    if any(kw in question_lower for kw in news_impact_keywords):
        return "news_impact"
    
    # 2. TRADE_EXPLAIN: Trading history queries - MUST depend on trade_history
    trade_explain_keywords = [
        'trading history', 'trade history', 'why buy', 'why sell',
        'my trades', 'my trading', 'trading actions', 'trading decisions',
        'buy sell', 'buying selling', 'trading record', 'transaction history',
        'what trades', 'recent trades', 'trading activity'
    ]
    if any(kw in question_lower for kw in trade_explain_keywords):
        return "trade_explain"
    
    # 3. MARKET_STATE: Trend and market state queries
    trend_keywords = [
        'trend', 'trending', 'trends',
        'up', 'down', 'upward', 'downward', 'upward trend', 'downward trend',
        'rising', 'falling', 'going up', 'going down', 'has been rising', 
        'has been falling', 'is rising', 'is falling', 'price movement',
        'price trend', 'market trend', 'stock trend', 'market state',
        'price', 'performance', 'how is', 'market performance'
    ]
    if any(kw in question_lower for kw in trend_keywords):
        return "market_state"
    
    # 4. RISK_CHECK: Risk-related queries
    if any(kw in question_lower for kw in ['risk', 'warning', 'danger', 'concern', 'risky']):
        return "risk_check"
    
    # 5. STRATEGY_SUGGEST: Strategy and recommendation queries
    if any(kw in question_lower for kw in ['strategy', 'suggest', 'recommend', 'advice', 'should i']):
        return "strategy_suggest"
    
    # Default to market_state (most common for general queries)
    return "market_state"


def infer_days_from_question(question: str) -> Optional[int]:
    """
    Infer number of days from question text
    
    Args:
        question: User question
        
    Returns:
        Number of days if found, None otherwise
    """
    q = question.lower()
    
    # Pattern: "last N days" or "past N days"
    m = re.search(r'(?:last|past)\s+(\d+)\s+days?', q)
    if m:
        return int(m.group(1))
    
    # Pattern: "N days" (when context suggests time window)
    m = re.search(r'(\d+)\s+days?', q)
    if m:
        days = int(m.group(1))
        # Only accept if it's a reasonable time window (1-365 days)
        if 1 <= days <= 365:
            return days
    
    # Common phrases
    if "past month" in q or "last month" in q:
        return 30
    if "past week" in q or "last week" in q:
        return 7
    if "past year" in q or "last year" in q:
        return 365
    if "past quarter" in q or "last quarter" in q:
        return 90
    
    return None


def _get_time_window(task_type: TaskType, question: str = "") -> int:
    """
    Get time window based on task type and question text (in days)
    
    Args:
        task_type: Task type
        question: User question (optional, for inferring days)
        
    Returns:
        Number of days for time window
    """
    # Try to infer from question first
    inferred_days = infer_days_from_question(question)
    if inferred_days is not None:
        logger.info(f"Inferred {inferred_days} days from question: '{question[:50]}...'")
        return inferred_days
    
    # Default windows based on task type
    windows = {
        "market_state": 60,      # 60 days market state (increased for trend analysis)
        "trade_explain": 90,     # 90 days trade history
        "risk_check": 60,        # 60 days risk check
        "news_impact": 14,       # 14 days news impact
        "strategy_suggest": 30,  # 30 days strategy suggestion
    }
    default_days = windows.get(task_type, 60)
    logger.debug(f"Using default time window for {task_type}: {default_days} days")
    return default_days


def _build_retrieval_needs(task_type: TaskType, request: RagRequest) -> Dict[DocType, RetrievalNeed]:
    """Build retrieval needs"""
    needs = {}
    
    # Enable different document types based on task type
    if task_type == "market_state":
        needs["trends"] = RetrievalNeed(enable=True, recall_k=50, final_k=10)
        needs["news_piece"] = RetrievalNeed(enable=True, recall_k=20, final_k=5)
        needs["trade_history"] = RetrievalNeed(enable=False, recall_k=0, final_k=0)
    elif task_type == "trade_explain":
        needs["trends"] = RetrievalNeed(enable=True, recall_k=30, final_k=5)
        needs["news_piece"] = RetrievalNeed(enable=False, recall_k=0, final_k=0)
        needs["trade_history"] = RetrievalNeed(enable=True, recall_k=100, final_k=20)
    elif task_type == "risk_check":
        needs["trends"] = RetrievalNeed(enable=True, recall_k=60, final_k=15)
        needs["news_piece"] = RetrievalNeed(enable=True, recall_k=30, final_k=10)
        needs["trade_history"] = RetrievalNeed(enable=True, recall_k=50, final_k=10)
    elif task_type == "news_impact":
        needs["trends"] = RetrievalNeed(enable=True, recall_k=20, final_k=5)
        needs["news_piece"] = RetrievalNeed(enable=True, recall_k=50, final_k=15)
        needs["trade_history"] = RetrievalNeed(enable=False, recall_k=0, final_k=0)
    elif task_type == "strategy_suggest":
        needs["trends"] = RetrievalNeed(enable=True, recall_k=40, final_k=10)
        needs["news_piece"] = RetrievalNeed(enable=True, recall_k=30, final_k=8)
        needs["trade_history"] = RetrievalNeed(enable=True, recall_k=30, final_k=8)
    else:
        # Default configuration
        needs["trends"] = RetrievalNeed(enable=True, recall_k=30, final_k=10)
        needs["news_piece"] = RetrievalNeed(enable=True, recall_k=20, final_k=5)
        needs["trade_history"] = RetrievalNeed(enable=True, recall_k=20, final_k=5)
    
    return needs


def _build_constraints(task_type: TaskType, request: RagRequest) -> Dict:
    """Build constraints"""
    constraints = {
        "frequency": request.frequency,
    }
    
    if task_type == "news_impact":
        constraints["min_impact"] = 3  # At least medium impact
    elif task_type == "risk_check":
        constraints["min_impact"] = 5  # High risk news
        constraints["check_volatility"] = True
    elif task_type == "trade_explain":
        constraints["max_trades"] = 100  # Maximum 100 trades
    
    return constraints


def _build_semantic_queries(question: str, task_type: TaskType) -> List[str]:
    """Build semantic queries (for news retrieval)"""
    queries = [question]  # Original question
    
    # Add expanded queries based on task type
    if task_type == "news_impact":
        queries.append("financial news impact")
        queries.append("stock market news")
    elif task_type == "risk_check":
        queries.append("risk warning news")
        queries.append("negative market news")
    
    return queries


def _build_constraint_queries(request: RagRequest, task_type: TaskType) -> List[str]:
    """Build constraint queries (for SQL filtering)"""
    queries = []
    
    if request.stock_code:
        queries.append(f"stock_code={request.stock_code}")
    
    queries.append(f"time_range=[{request.decision_time}]")
    
    if task_type == "news_impact":
        queries.append("impact>=3")
    elif task_type == "risk_check":
        queries.append("impact>=5")
    
    return queries

