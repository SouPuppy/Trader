"""
Post-Verifier
Reference validation, time validation, boundary validation, confidence gating
"""
import re
from datetime import datetime
from typing import List
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from trader.rag.types import VerifiedAnswer, EvidencePack
from trader.rag.normalize_citations import normalize_citations, extract_citations
from trader.rag import get_rag_logger

logger = get_rag_logger(__name__)

# Standard citation format: [DOC:type:stock:date]
CITATION_PATTERN = re.compile(r'\[DOC:([^:]+):([^:]+):([^\]]+)\]')


def verify_answer(raw_answer: str, evidence: EvidencePack) -> VerifiedAnswer:
    """
    Verify answer
    
    Args:
        raw_answer: Raw answer
        evidence: Evidence pack
        
    Returns:
        VerifiedAnswer
    """
    violations = []
    
    # Step 1: Normalize citations (convert old formats, reject invalid ones)
    normalized_answer, citation_violations = normalize_citations(raw_answer)
    violations.extend(citation_violations)
    
    # Step 2: Extract ONLY [DOC:type:stock:date] format citations
    # NO MORE guessing from natural language
    citation_strings = extract_citations(normalized_answer)
    
    # Step 3: Convert citations to doc_ids for validation
    # Build mapping from citation format to doc_id
    citation_to_doc_id = {}
    for item in evidence.items:
        # Extract date from timestamp (YYYY-MM-DD format)
        timestamp_date = item.timestamp[:10] if len(item.timestamp) >= 10 else item.timestamp
        citation_format = f"[DOC:{item.doc_type}:{item.stock_code}:{timestamp_date}]"
        citation_to_doc_id[citation_format] = item.doc_id
    
    # Convert citations to doc_ids
    used_doc_ids = []
    invalid_citations = []
    for citation in citation_strings:
        if citation in citation_to_doc_id:
            used_doc_ids.append(citation_to_doc_id[citation])
        else:
            invalid_citations.append(citation)
    
    if invalid_citations:
        violations.append(f"Referenced non-existent citations: {invalid_citations}")
    
    # Validate referenced doc_ids exist in evidence pack
    valid_doc_ids = {item.doc_id for item in evidence.items}
    
    # Check if all used doc_ids are valid (no fuzzy matching - strict validation)
    invalid_refs = []
    for doc_id in used_doc_ids:
        if doc_id not in valid_doc_ids:
            invalid_refs.append(doc_id)
    
    if invalid_refs:
        violations.append(f"Referenced non-existent doc_id: {invalid_refs}")
    
    # Time validation: check if referenced document time is before decision time
    try:
        decision_time = datetime.fromisoformat(evidence.decision_time.replace('Z', '+00:00'))
        
        for doc_id in used_doc_ids:
            # Find corresponding evidence item
            item = next((item for item in evidence.items if item.doc_id == doc_id), None)
            if item:
                try:
                    # Extract date part for comparison
                    item_timestamp = item.timestamp[:10] if len(item.timestamp) >= 10 else item.timestamp
                    item_time = datetime.fromisoformat(item_timestamp)
                    decision_date = decision_time.date()
                    item_date = item_time.date()
                    
                    if item_date > decision_date:
                        violations.append(f"Document citation time ({item_timestamp}) is after decision time ({evidence.decision_time[:10]})")
                except (ValueError, AttributeError) as e:
                    logger.debug(f"Time validation error for {doc_id}: {e}")
                    pass
    except (ValueError, AttributeError):
        logger.warning("Cannot parse decision time, skipping time validation")
    
    # Confidence gating: check total evidence score
    # Use different thresholds based on task type
    # Market state/trend questions may need more historical data, so use lower threshold
    total_score = sum(item.relevance_score for item in evidence.items)
    
    # Adjust threshold based on task type
    if evidence.task_type == "market_state":
        # Trend analysis needs more data points, so lower threshold
        min_score_threshold = 0.5
    elif evidence.task_type == "trade_explain":
        # Trade history may have many low-scoring items
        min_score_threshold = 1.0
    elif evidence.task_type == "risk_check":
        # Risk check needs high-quality evidence
        min_score_threshold = 1.5
    else:
        # Default threshold for other task types
        min_score_threshold = 2.0
    
    if total_score < min_score_threshold:
        violations.append(f"Insufficient total evidence score ({total_score:.2f} < {min_score_threshold})")
    
    # Check if answer contains key facts outside evidence pack
    # Simplified: check if explicitly states "insufficient evidence"
    has_insufficient_evidence = any(
        phrase in raw_answer.lower() 
        for phrase in ["insufficient evidence", "cannot answer", "not enough evidence"]
    )
    
    # Determine if verification passed
    passed = len(violations) == 0
    
    # Determine mode
    if passed and not has_insufficient_evidence:
        mode = "normal"
    else:
        mode = "degraded"
    
    # Use normalized answer (with normalized citations)
    # Add warning prefix if there are violations
    if violations:
        warning_prefix = "[Warning]: The following issues may make the answer unreliable:\n" + "\n".join(f"- {v}" for v in violations) + "\n\n"
        answer = warning_prefix + normalized_answer
    else:
        answer = normalized_answer
    
    verified = VerifiedAnswer(
        answer=answer,
        passed=passed,
        violations=violations,
        used_doc_ids=used_doc_ids,
        mode=mode
    )
    
    logger.info(f"Answer verification completed: passed={passed}, mode={mode}, violations={len(violations)}")
    
    return verified

