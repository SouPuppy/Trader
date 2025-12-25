"""
Evidence Builder
Convert reranked Candidates to EvidenceItems (compress, extract key_facts, unify signals)
"""
from typing import Dict, List
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from trader.rag.types import RetrievalPlan, Candidate, EvidenceItem, EvidencePack, DocType
from trader.news.analyze import analyze
from trader.rag import get_rag_logger

logger = get_rag_logger(__name__)


def build_evidence(plan: RetrievalPlan, ranked: Dict[DocType, List[Candidate]]) -> EvidencePack:
    """
    Build evidence pack
    
    Args:
        plan: Retrieval plan
        ranked: Ranked candidates (grouped by type)
        
    Returns:
        EvidencePack
    """
    items = []
    
    # Process candidates by type
    for doc_type, candidates in ranked.items():
        for candidate in candidates:
            try:
                evidence_item = _candidate_to_evidence(candidate, doc_type, plan)
                if evidence_item:
                    items.append(evidence_item)
            except Exception as e:
                logger.warning(f"Failed to build evidence item (doc_id={candidate.doc.doc_id}): {e}")
                continue
    
    # Sort by relevance_score
    items.sort(key=lambda x: x.relevance_score, reverse=True)
    
    evidence_pack = EvidencePack(
        task_type=plan.task_type,
        stock_code=plan.stock_code,
        decision_time=plan.time_end,
        items=items
    )
    
    logger.info(f"Built evidence pack: {len(items)} evidence items")
    
    return evidence_pack


def _candidate_to_evidence(
    candidate: Candidate, 
    doc_type: DocType, 
    plan: RetrievalPlan
) -> EvidenceItem:
    """Convert Candidate to EvidenceItem"""
    doc = candidate.doc
    payload = doc.payload
    
    # Extract key_facts
    key_facts = _extract_key_facts(candidate, doc_type, plan)
    
    # Extract signals
    signals = _extract_signals(candidate, doc_type, plan)
    
    # Use reranked score (simplified: using recall_score)
    # Should actually use final_score calculated in rerank stage
    relevance_score = candidate.recall_score
    
    evidence_item = EvidenceItem(
        doc_id=doc.doc_id,
        doc_type=doc_type,
        stock_code=doc.stock_code,
        timestamp=doc.timestamp,
        key_facts=key_facts,
        signals=signals,
        relevance_score=relevance_score
    )
    
    return evidence_item


def _extract_key_facts(candidate: Candidate, doc_type: DocType, plan: RetrievalPlan) -> List[str]:
    """Extract key facts"""
    doc = candidate.doc
    payload = doc.payload
    facts = []
    
    if doc_type == "news_piece":
        # news: use paraphrase directly as a key_fact
        paraphrase = payload.get("paraphrase")
        if paraphrase:
            facts.append(paraphrase)
        else:
            # If no paraphrase, try to extract from text
            text = doc.text or ""
            if text:
                # Take first 150 characters as summary
                preview = text[:150].strip()
                if preview:
                    facts.append(preview)
        
        # Add sentiment/impact information
        sentiment = payload.get("sentiment")
        impact = payload.get("impact")
        if sentiment is not None or impact is not None:
            fact_parts = []
            if sentiment is not None:
                fact_parts.append(f"Sentiment: {sentiment}")
            if impact is not None:
                fact_parts.append(f"Impact: {impact}")
            if fact_parts:
                facts.append("; ".join(fact_parts))
    
    elif doc_type == "trade_history":
        # trade: extract trade action
        action = payload.get("action", "")
        price = payload.get("price", 0)
        volume = payload.get("volume", 0)
        
        if action and price and volume:
            facts.append(f"{action} {volume} shares @ {price:.2f}")
        
        # Can add more aggregated information (e.g., recent N trades, average price)
        # Simplified here
    
    elif doc_type == "trends":
        # trends: extract core anomaly signals
        signal_parts = []
        
        ret_5d = payload.get("ret_5d")
        if ret_5d is not None:
            signal_parts.append(f"5d return: {ret_5d:.2%}")
        
        vol_z_20d = payload.get("vol_z_20d")
        if vol_z_20d is not None:
            signal_parts.append(f"20d vol Z-score: {vol_z_20d:.2f}")
        
        gap_pct = payload.get("gap_pct")
        if gap_pct is not None:
            signal_parts.append(f"Gap pct: {gap_pct:.2%}")
        
        if signal_parts:
            facts.append("; ".join(signal_parts))
        else:
            facts.append("Trend feature data")
    
    # Ensure at least 1 fact
    if not facts:
        facts.append(f"{doc_type} data")
    
    return facts[:3]  # Maximum 3 facts


def _extract_signals(candidate: Candidate, doc_type: DocType, plan: RetrievalPlan) -> Dict:
    """Extract numerical signals"""
    doc = candidate.doc
    payload = doc.payload
    signals = {}
    
    if doc_type == "news_piece":
        # news signals
        sentiment = payload.get("sentiment")
        impact = payload.get("impact")
        
        if sentiment is not None:
            signals["sentiment"] = sentiment
        if impact is not None:
            signals["impact"] = impact
    
    elif doc_type == "trends":
        # trends signals: extract key features
        for key in ["ret_1d", "ret_5d", "ret_20d", "vol_z_20d", "gap_pct", 
                    "pe_ratio", "pb_ratio", "ps_ratio"]:
            value = payload.get(key)
            if value is not None:
                signals[key] = value
    
    elif doc_type == "trade_history":
        # trade signals
        signals["action"] = payload.get("action", "")
        signals["price"] = payload.get("price", 0)
        signals["volume"] = payload.get("volume", 0)
    
    return signals

