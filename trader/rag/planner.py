"""
Retrieval Planner
Generate retrieval plan JSON, decide which doc_types, time windows, k values, constraints and query forms
"""
from datetime import datetime, timedelta
from typing import Dict, List
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
    
    # Set time window based on task type
    time_window_days = _get_time_window(task_type)
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
    """Infer task type from question"""
    question_lower = question.lower()
    
    # Trend-related keywords (check first as they are more specific)
    trend_keywords = [
        'trend', 'trending', 'trends',
        'up', 'down', 'upward', 'downward', 'upward trend', 'downward trend',
        'rising', 'falling', 'going up', 'going down', 'has been rising', 
        'has been falling', 'is rising', 'is falling', 'price movement',
        'price trend', 'market trend', 'stock trend'
    ]
    if any(kw in question_lower for kw in trend_keywords):
        return "market_state"
    
    # Market state keywords
    if any(kw in question_lower for kw in ['market state', 'price', 'performance', 'how is']):
        return "market_state"
    elif any(kw in question_lower for kw in ['trade', 'buy', 'sell', 'history', 'transaction']):
        return "trade_explain"
    elif any(kw in question_lower for kw in ['risk', 'warning', 'danger', 'concern']):
        return "risk_check"
    elif any(kw in question_lower for kw in ['news', 'event', 'announcement']):
        return "news_impact"
    elif any(kw in question_lower for kw in ['strategy', 'suggest', 'recommend', 'advice']):
        return "strategy_suggest"
    else:
        # Default to market_state (most common for general queries)
        return "market_state"


def _get_time_window(task_type: TaskType) -> int:
    """Get time window based on task type (in days)"""
    windows = {
        "market_state": 60,      # 60 days market state (increased for trend analysis)
        "trade_explain": 90,     # 90 days trade history
        "risk_check": 60,        # 60 days risk check
        "news_impact": 14,       # 14 days news impact
        "strategy_suggest": 30,  # 30 days strategy suggestion
    }
    return windows.get(task_type, 60)


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

