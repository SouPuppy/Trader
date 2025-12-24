"""
Reranker (Time Decay + Relevance + Diversity)
Select final_k from recall_k candidates, output final ranked results
"""
from datetime import datetime
from typing import Dict, List
import sys
from pathlib import Path
import math

project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from trader.rag.types import RetrievalPlan, Candidate, DocType
from trader.logger import get_logger

logger = get_logger(__name__)


def rerank(plan: RetrievalPlan, candidates: List[Candidate]) -> Dict[DocType, List[Candidate]]:
    """
    Rerank candidates
    
    Args:
        plan: Retrieval plan
        candidates: Merged candidate list
        
    Returns:
        Ranked candidates grouped by type
    """
    # Group by type
    by_type: Dict[DocType, List[Candidate]] = {
        "trends": [],
        "news_piece": [],
        "trade_history": []
    }
    
    for candidate in candidates:
        doc_type = candidate.doc.doc_type
        if doc_type in by_type:
            by_type[doc_type].append(candidate)
    
    # Rerank each type
    ranked: Dict[DocType, List[Candidate]] = {}
    
    for doc_type, cands in by_type.items():
        need = plan.needs.get(doc_type)
        if not need or not need.enable:
            ranked[doc_type] = []
            continue
        
        # Calculate rerank scores
        scored_cands = []
        for cand in cands:
            rel_score = _calculate_relevance_score(cand, plan)
            time_decay = _calculate_time_decay(cand, plan)
            quality = _calculate_quality_score(cand, doc_type)
            
            # Combined score
            final_score = 0.4 * rel_score + 0.3 * time_decay + 0.3 * quality
            
            # Create new Candidate (update score)
            # Note: Candidate is frozen dataclass, cannot modify directly
            # We keep original recall_score here, use final_score in evidence stage
            scored_cands.append((cand, final_score))
        
        # Sort by score
        scored_cands.sort(key=lambda x: x[1], reverse=True)
        
        # Diversity filter (for news_piece)
        if doc_type == "news_piece":
            scored_cands = _apply_diversity_filter(scored_cands, plan)
        
        # Take top-k
        final_cands = [cand for cand, _ in scored_cands[:need.final_k]]
        ranked[doc_type] = final_cands
        
        logger.info(f"{doc_type} rerank: {len(cands)} -> {len(final_cands)}")
    
    return ranked


def _calculate_relevance_score(candidate: Candidate, plan: RetrievalPlan) -> float:
    """Calculate relevance score (normalized recall score)"""
    # Simple normalization: assume recall score in [0, 1] range
    return min(1.0, max(0.0, candidate.recall_score))


def _calculate_time_decay(candidate: Candidate, plan: RetrievalPlan) -> float:
    """Calculate time decay score"""
    try:
        doc_time = datetime.fromisoformat(candidate.doc.timestamp.replace('Z', '+00:00'))
        plan_time = datetime.fromisoformat(plan.time_end.replace('Z', '+00:00'))
        days_diff = abs((plan_time - doc_time).days)
        
        # Time decay: exp(-days/τ), τ adjusted by doc_type
        tau = 7 if candidate.doc.doc_type == "news_piece" else 30
        time_score = math.exp(-days_diff / tau)
        
        return time_score
    except Exception:
        return 0.5


def _calculate_quality_score(candidate: Candidate, doc_type: DocType) -> float:
    """Calculate quality score"""
    payload = candidate.doc.payload
    
    if doc_type == "news_piece":
        # news quality: based on impact (if available)
        impact = payload.get("impact", 0)
        if isinstance(impact, (int, float)):
            return min(1.0, impact / 10.0)
        return 0.5
    
    elif doc_type == "trends":
        # trends quality: based on data completeness
        feature_count = sum(1 for k, v in payload.items() 
                           if k != "frequency" and v is not None)
        max_features = 15  # Expected feature count
        return min(1.0, feature_count / max_features)
    
    elif doc_type == "trade_history":
        # trade quality: based on volume
        volume = payload.get("volume", 0)
        if isinstance(volume, (int, float)) and volume > 0:
            # Normalize: assume max volume is 10000
            return min(1.0, math.log10(volume + 1) / 5.0)
        return 0.5
    
    return 0.5


def _apply_diversity_filter(
    scored_cands: List[tuple], 
    plan: RetrievalPlan
) -> List[tuple]:
    """
    Apply diversity filter (for news_piece)
    Simple semantic deduplication: based on text similarity threshold
    """
    if len(scored_cands) <= 1:
        return scored_cands
    
    # Simple diversity filter: if text similarity too high, keep only highest score
    filtered = []
    seen_texts = []
    
    for cand, score in scored_cands:
        text = cand.doc.text or ""
        text_preview = text[:100]  # Use first 100 chars as fingerprint
        
        # Check if similar to existing texts
        is_similar = False
        for seen_text in seen_texts:
            # Simple similarity check: if first 100 chars same, consider duplicate
            if text_preview == seen_text[:100]:
                is_similar = True
                break
        
        if not is_similar:
            filtered.append((cand, score))
            seen_texts.append(text_preview)
    
    logger.debug(f"Diversity filter: {len(scored_cands)} -> {len(filtered)}")
    return filtered

