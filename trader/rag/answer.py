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
from .evidence_gate import apply_evidence_gate, check_data_coverage
from .prompt import build_prompt
from .generate import llm_generate
from .normalize_citations import normalize_citations, sanitize_citations
from .verify import verify_answer
from .observability import log_all
from .db.queries import ensure_tables
from trader.rag import get_rag_logger

logger = get_rag_logger(__name__)


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
    
    # 2.5. Check data coverage (preflight check)
    coverage_msg = check_data_coverage(plan)
    if coverage_msg:
        logger.warning(f"Data coverage check failed: {coverage_msg}")
        # Continue with retrieval, but will degrade later
    
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
    
    # 6.5. Apply evidence gate (check if required evidence exists)
    evidence, degradation_msg = apply_evidence_gate(evidence)
    
    # 6.6. Combine coverage and evidence degradation messages
    if coverage_msg:
        if degradation_msg:
            degradation_msg = f"{coverage_msg}\n{degradation_msg}"
        else:
            degradation_msg = coverage_msg
    
    # 7. Build prompt (include degradation message if any)
    prompt = build_prompt(req, evidence, degradation_msg=degradation_msg)
    
    # 8. LLM generation
    raw = llm_generate(prompt)
    
    # 8.5. Normalize citations before verification
    normalized_raw, citation_violations = normalize_citations(raw)
    if citation_violations:
        logger.warning(f"Citation normalization violations: {citation_violations}")
    
    # 8.6. Sanitize citations (remove citations not in evidence pack)
    allowed_doc_ids = {item.doc_id for item in evidence.items}
    sanitized_raw = sanitize_citations(normalized_raw, allowed_doc_ids, evidence.items)
    
    # If evidence is empty, force degraded mode
    if len(evidence.items) == 0:
        logger.warning("Evidence pack is empty, forcing degraded mode")
        # Don't generate answer with citations if no evidence
        if not degradation_msg:
            degradation_msg = "Insufficient evidence available. Cannot provide answer with citations."
    
    # 9. Verify answer (with sanitized citations)
    verified = verify_answer(sanitized_raw, evidence)
    
    # 10. Log all information
    log_all(req, plan, ranked, evidence, verified)
    
    return verified

