"""
RAG System End-to-End Interface
Chain modules together
"""
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from .types import VerifiedAnswer
from .normalize import normalize_request
from .planner import build_plan
from .retrievers import NewsRetriever, TradeRetriever, TrendsRetriever
from .merger import merge_candidates
from .rerank import rerank
from .evidence import build_evidence
from .prompt import build_prompt
from .generate import llm_generate
from .verify import verify_answer
from .observability import log_all
from .db.queries import ensure_tables
from trader.logger import get_logger

logger = get_logger(__name__)


def rag_answer(
    question: str,
    stock_code: str = None,
    decision_time: str = None,
    frequency: str = "1d",
    context: dict = None
) -> VerifiedAnswer:
    """
    RAG System End-to-End Interface
    
    Args:
        question: User question
        stock_code: Stock code (optional)
        decision_time: Decision time (ISO8601, optional)
        frequency: Frequency ('1d'/'1h'...)
        context: Context (optional)
        
    Returns:
        VerifiedAnswer
    """
    # Ensure database tables exist
    ensure_tables()
    
    # 1. Request normalization
    req = normalize_request(question, stock_code, decision_time, frequency, context)
    
    # 2. Build retrieval plan
    plan = build_plan(req)
    
    # 3. Retrieve candidates by type
    news_retriever = NewsRetriever()
    trade_retriever = TradeRetriever()
    trends_retriever = TrendsRetriever()
    
    news_cands = news_retriever.retrieve(plan)
    trade_cands = trade_retriever.retrieve(plan)
    trends_cands = trends_retriever.retrieve(plan)
    
    # 4. Merge candidates
    merged = merge_candidates({
        "news_piece": news_cands,
        "trade_history": trade_cands,
        "trends": trends_cands
    })
    
    # 5. Rerank
    ranked = rerank(plan, merged)
    
    # 6. Build evidence pack
    evidence = build_evidence(plan, ranked)
    
    # 7. Build prompt
    prompt = build_prompt(req, evidence)
    
    # 8. LLM generation
    raw = llm_generate(prompt)
    
    # 9. Verify answer
    verified = verify_answer(raw, evidence)
    
    # 10. Log all information
    log_all(req, plan, ranked, evidence, verified)
    
    return verified

