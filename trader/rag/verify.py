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
from trader.logger import get_logger

logger = get_logger(__name__)


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
    used_doc_ids = []
    
    # Extract referenced doc_ids
    # Pattern: doc_id: <id> or doc_id <id>
    # Doc IDs can contain colons, spaces, and other characters (e.g., "news:4000:2023-12-29 00:00:00")
    # Try multiple patterns to catch different formats
    used_doc_ids = []
    
    # Pattern 1: doc_id: <id> (match until end of line, closing paren, or punctuation)
    pattern1 = r'doc_id[:\s]+([^\n\)\],;.]+?)(?:\s*[.,;\)\]]|\s*$|\n)'
    matches1 = re.findall(pattern1, raw_answer, re.IGNORECASE)
    used_doc_ids.extend([m.strip() for m in matches1 if m.strip()])
    
    # Pattern 2: (doc_id: <id>) - match inside parentheses
    pattern2 = r'\(doc_id[:\s]+([^\)]+)\)'
    matches2 = re.findall(pattern2, raw_answer, re.IGNORECASE)
    used_doc_ids.extend([m.strip() for m in matches2 if m.strip()])
    
    # Deduplicate
    used_doc_ids = list(set(used_doc_ids))
    
    # Validate referenced doc_ids exist in evidence pack
    valid_doc_ids = {item.doc_id for item in evidence.items}
    
    # Try fuzzy matching: if exact match fails, try matching without time part
    invalid_refs = []
    matched_doc_ids = set()
    
    for doc_id in used_doc_ids:
        if doc_id in valid_doc_ids:
            matched_doc_ids.add(doc_id)
        else:
            # Try fuzzy match: remove time part (everything after last space or colon)
            # e.g., "news:4000:2023-12-29 00:00:00" -> try "news:4000:2023-12-29"
            fuzzy_match = None
            # Try removing time part (after last space)
            if ' ' in doc_id:
                base_id = doc_id.rsplit(' ', 1)[0]
                if base_id in valid_doc_ids:
                    fuzzy_match = base_id
            # Try removing everything after last colon (if it looks like a time)
            if not fuzzy_match and doc_id.count(':') >= 2:
                parts = doc_id.rsplit(':', 1)
                if len(parts) == 2 and parts[1].replace(' ', '').replace('-', '').isdigit():
                    base_id = parts[0]
                    if base_id in valid_doc_ids:
                        fuzzy_match = base_id
            
            if fuzzy_match:
                matched_doc_ids.add(fuzzy_match)
                logger.debug(f"Fuzzy matched doc_id: '{doc_id}' -> '{fuzzy_match}'")
            else:
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
                    item_time = datetime.fromisoformat(item.timestamp.replace('Z', '+00:00'))
                    if item_time > decision_time:
                        violations.append(f"Document {doc_id} time ({item.timestamp}) is after decision time ({evidence.decision_time})")
                except (ValueError, AttributeError):
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
    
    # Add warning prefix if there are violations
    if violations:
        warning_prefix = "[Warning]: The following issues may make the answer unreliable:\n" + "\n".join(f"- {v}" for v in violations) + "\n\n"
        answer = warning_prefix + raw_answer
    else:
        answer = raw_answer
    
    verified = VerifiedAnswer(
        answer=answer,
        passed=passed,
        violations=violations,
        used_doc_ids=used_doc_ids,
        mode=mode
    )
    
    logger.info(f"Answer verification completed: passed={passed}, mode={mode}, violations={len(violations)}")
    
    return verified

