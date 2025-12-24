"""
Observability
Log request / plan / recall counts / rerank score distribution / EvidencePack / verifier results
"""
from typing import Dict, Any, List
import sys
from pathlib import Path
import json

project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from trader.rag.types import (
    RagRequest, RetrievalPlan, Candidate, EvidencePack, VerifiedAnswer, DocType
)
from trader.logger import get_logger

logger = get_logger(__name__)


def log_all(
    request: RagRequest,
    plan: RetrievalPlan,
    ranked: Dict[DocType, List[Candidate]],
    evidence: EvidencePack,
    verified: VerifiedAnswer
):
    """
    Log all key information
    
    Args:
        request: RAG request
        plan: Retrieval plan
        ranked: Ranked candidates
        evidence: Evidence pack
        verified: Verified answer
    """
    # Log request
    logger.info("=" * 80)
    logger.info("RAG Request")
    logger.info("=" * 80)
    logger.info(f"Question: {request.question}")
    logger.info(f"Stock Code: {request.stock_code}")
    logger.info(f"Decision Time: {request.decision_time}")
    logger.info(f"Frequency: {request.frequency}")
    
    # Log plan
    logger.info("=" * 80)
    logger.info("Retrieval Plan")
    logger.info("=" * 80)
    logger.info(f"Task Type: {plan.task_type}")
    logger.info(f"Time Window: [{plan.time_start}, {plan.time_end}]")
    logger.info(f"Retrieval Needs:")
    for doc_type, need in plan.needs.items():
        if need.enable:
            logger.info(f"  {doc_type}: recall_k={need.recall_k}, final_k={need.final_k}")
    logger.info(f"Constraints: {plan.constraints}")
    
    # Log recall counts
    logger.info("=" * 80)
    logger.info("Recall Statistics")
    logger.info("=" * 80)
    for doc_type, candidates in ranked.items():
        logger.info(f"{doc_type}: {len(candidates)} candidates")
    
    # Log rerank score distribution
    logger.info("=" * 80)
    logger.info("Rerank Score Distribution")
    logger.info("=" * 80)
    for doc_type, candidates in ranked.items():
        if candidates:
            scores = [c.recall_score for c in candidates]
            min_score = min(scores)
            max_score = max(scores)
            mean_score = sum(scores) / len(scores)
            logger.info(f"{doc_type}: min={min_score:.3f}, mean={mean_score:.3f}, max={max_score:.3f}")
    
    # Log evidence pack (can be desensitized)
    logger.info("=" * 80)
    logger.info("Evidence Pack")
    logger.info("=" * 80)
    logger.info(f"Task Type: {evidence.task_type}")
    logger.info(f"Evidence Item Count: {len(evidence.items)}")
    logger.info("Evidence Item Summary:")
    for i, item in enumerate(evidence.items[:5], 1):  # Show first 5 only
        logger.info(f"  {i}. [{item.doc_type}] {item.doc_id}: "
                   f"score={item.relevance_score:.3f}, "
                   f"key_facts={len(item.key_facts)} items")
    
    # Log verification results
    logger.info("=" * 80)
    logger.info("Verification Results")
    logger.info("=" * 80)
    logger.info(f"Passed: {verified.passed}")
    logger.info(f"Mode: {verified.mode}")
    logger.info(f"Used doc_ids: {verified.used_doc_ids}")
    if verified.violations:
        logger.warning(f"Violations ({len(verified.violations)}):")
        for violation in verified.violations:
            logger.warning(f"  - {violation}")
    
    logger.info("=" * 80)


def log_metrics(
    request: RagRequest,
    plan: RetrievalPlan,
    ranked: Dict[DocType, List[Candidate]],
    evidence: EvidencePack,
    verified: VerifiedAnswer
) -> Dict[str, Any]:
    """
    Generate metrics dictionary (for monitoring/analysis)
    
    Returns:
        Metrics dictionary
    """
    metrics = {
        "request": {
            "question_length": len(request.question),
            "has_stock_code": request.stock_code is not None,
            "frequency": request.frequency
        },
        "plan": {
            "task_type": plan.task_type,
            "time_window_days": _calculate_days(plan.time_start, plan.time_end),
            "enabled_doc_types": [dt for dt, need in plan.needs.items() if need.enable]
        },
        "recall": {
            doc_type: len(candidates) 
            for doc_type, candidates in ranked.items()
        },
        "rerank_scores": {
            doc_type: {
                "min": min([c.recall_score for c in candidates]) if candidates else 0,
                "mean": sum([c.recall_score for c in candidates]) / len(candidates) if candidates else 0,
                "max": max([c.recall_score for c in candidates]) if candidates else 0,
                "count": len(candidates)
            }
            for doc_type, candidates in ranked.items()
        },
        "evidence": {
            "item_count": len(evidence.items),
            "total_score": sum(item.relevance_score for item in evidence.items),
            "by_type": {
                doc_type: len([item for item in evidence.items if item.doc_type == doc_type])
                for doc_type in ["trends", "news_piece", "trade_history"]
            }
        },
        "verification": {
            "passed": verified.passed,
            "mode": verified.mode,
            "violation_count": len(verified.violations),
            "used_doc_count": len(verified.used_doc_ids)
        }
    }
    
    return metrics


def _calculate_days(time_start: str, time_end: str) -> float:
    """Calculate time window in days"""
    try:
        from datetime import datetime
        start = datetime.fromisoformat(time_start.replace('Z', '+00:00'))
        end = datetime.fromisoformat(time_end.replace('Z', '+00:00'))
        return (end - start).days
    except Exception:
        return 0.0

