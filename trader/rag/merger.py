"""
Candidate Merger
Merge three types of candidates and deduplicate by doc_id
"""
from typing import Dict, List
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from trader.rag.types import Candidate, DocType
from trader.rag import get_rag_logger

logger = get_rag_logger(__name__)


def merge_candidates(cands_by_type: Dict[DocType, List[Candidate]]) -> List[Candidate]:
    """
    Merge candidate pool, deduplicate by doc_id
    
    Args:
        cands_by_type: Candidate list grouped by type
        
    Returns:
        Merged candidate list (deduplicated)
    """
    seen_doc_ids = set()
    merged = []
    
    # Merge by type order (maintain priority)
    type_order = ["trends", "news_piece", "trade_history"]
    
    for doc_type in type_order:
        candidates = cands_by_type.get(doc_type, [])
        for candidate in candidates:
            doc_id = candidate.doc.doc_id
            if doc_id not in seen_doc_ids:
                seen_doc_ids.add(doc_id)
                merged.append(candidate)
            else:
                logger.debug(f"Skipping duplicate document: {doc_id}")
    
    logger.info(f"Merged candidate pool: input {sum(len(cands) for cands in cands_by_type.values())} candidates, "
                f"after deduplication {len(merged)} candidates")
    
    return merged

