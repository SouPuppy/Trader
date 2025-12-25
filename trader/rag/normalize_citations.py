"""
Citation Normalizer
Normalize citations to [DOC:type:stock:date] format and reject invalid formats
"""
import re
from typing import List, Tuple
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from trader.rag import get_rag_logger

logger = get_rag_logger(__name__)

# Standard citation format: [DOC:type:stock:date]
CITATION_PATTERN = re.compile(r'\[DOC:([^:]+):([^:]+):([^\]]+)\]')


def normalize_citations(text: str) -> Tuple[str, List[str]]:
    """
    Normalize citations in text to [DOC:type:stock:date] format
    
    Args:
        text: Text containing citations
        
    Returns:
        Tuple of (normalized_text, violations)
        - normalized_text: Text with normalized citations
        - violations: List of violation messages for invalid citations
    """
    violations = []
    normalized_text = text
    
    # Find all [DOC:...] citations
    doc_citations = CITATION_PATTERN.findall(text)
    
    # Check for invalid patterns
    # 1. Comma-separated citations: [DOC:A, DOC:B]
    comma_pattern = re.compile(r'\[DOC:[^\]]+,\s*DOC:[^\]]+\]')
    comma_matches = comma_pattern.findall(text)
    if comma_matches:
        violations.append(f"Found comma-separated citations (forbidden): {comma_matches}")
        # Try to split them
        for match in comma_matches:
            # Extract individual citations
            parts = match.split(',')
            normalized_citations = []
            for part in parts:
                part = part.strip()
                if part.startswith('DOC:'):
                    part = '[' + part
                if not part.endswith(']'):
                    part = part + ']'
                # Validate format
                if CITATION_PATTERN.match(part):
                    normalized_citations.append(part)
            if normalized_citations:
                normalized_text = normalized_text.replace(match, ' '.join(normalized_citations))
    
    # 2. Range citations: [DOC:A to DOC:B] or [DOC:A through DOC:B]
    range_pattern = re.compile(r'\[DOC:[^\]]+\s+(?:to|through)\s+DOC:[^\]]+\]', re.IGNORECASE)
    range_matches = range_pattern.findall(text)
    if range_matches:
        violations.append(f"Found range citations (forbidden, will be rejected): {range_matches}")
        # Remove range citations (don't expand them)
        for match in range_matches:
            normalized_text = normalized_text.replace(match, '[INVALID_RANGE_CITATION_REMOVED]')
    
    # 3. Truncated IDs: [DOC:trends:AAPL] (missing date)
    truncated_pattern = re.compile(r'\[DOC:([^:]+):([^:\]]+)\]')
    truncated_matches = truncated_pattern.findall(text)
    for match in truncated_matches:
        doc_type, stock = match
        # Check if it's missing the date part (should have 3 parts: type:stock:date)
        if len(match) == 2:
            violations.append(f"Found truncated citation (missing date): [DOC:{doc_type}:{stock}]")
            # Don't auto-fix truncated citations, mark as invalid
            normalized_text = normalized_text.replace(
                f"[DOC:{doc_type}:{stock}]",
                f"[INVALID_TRUNCATED_CITATION:{doc_type}:{stock}]"
            )
    
    # 4. Old format: doc_id: trends:AAPL.O:2023-12-28
    old_format_pattern = re.compile(r'doc_id[:\s]+([^\s\)\],;.]+)', re.IGNORECASE)
    old_matches = old_format_pattern.findall(text)
    if old_matches:
        violations.append(f"Found old doc_id format (will convert): {old_matches}")
        for match in old_matches:
            # Try to parse and convert to new format
            parts = match.split(':')
            if len(parts) >= 3:
                doc_type = parts[0]
                stock = parts[1]
                date = parts[2][:10] if len(parts[2]) >= 10 else parts[2]  # Take date part only
                new_citation = f"[DOC:{doc_type}:{stock}:{date}]"
                normalized_text = re.sub(
                    rf'doc_id[:\s]+{re.escape(match)}',
                    new_citation,
                    normalized_text,
                    flags=re.IGNORECASE
                )
    
    # Extract all valid citations after normalization
    valid_citations = CITATION_PATTERN.findall(normalized_text)
    
    if violations:
        logger.warning(f"Citation normalization found violations: {violations}")
    
    return normalized_text, violations


def extract_citations(text: str) -> List[str]:
    """
    Extract all valid [DOC:type:stock:date] citations from text
    
    Args:
        text: Text containing citations
        
    Returns:
        List of citation strings in [DOC:type:stock:date] format
    """
    citations = CITATION_PATTERN.findall(text)
    return [f"[DOC:{doc_type}:{stock}:{date}]" for doc_type, stock, date in citations]


def validate_citation_format(citation: str) -> bool:
    """
    Validate if a citation matches [DOC:type:stock:date] format
    
    Args:
        citation: Citation string
        
    Returns:
        True if valid, False otherwise
    """
    return bool(CITATION_PATTERN.match(citation.strip()))


def sanitize_citations(text: str, allowed_doc_ids: set[str], evidence_items: list = None) -> str:
    """
    Sanitize citations by removing any citation not in the allowed doc_ids set
    
    This is a critical safety function to prevent fabricated citations.
    If evidence_pack is empty, all citations should be removed.
    
    Args:
        text: Text containing citations
        allowed_doc_ids: Set of allowed doc_ids (from evidence pack)
        evidence_items: Optional list of EvidenceItem objects for better matching
        
    Returns:
        Sanitized text with only valid citations
    """
    if not allowed_doc_ids:
        # If no evidence, remove all citations
        cleaned = CITATION_PATTERN.sub('', text)
        # Collapse repeated whitespace from removals
        cleaned = re.sub(r'[ \t]+', ' ', cleaned)
        cleaned = re.sub(r'\n{3,}', '\n\n', cleaned)
        return cleaned.strip()
    
    # Build mapping from citation format to doc_id
    # Citation format: [DOC:type:stock:date]
    # We need to match this to doc_ids in the evidence pack
    
    # Build a set of valid citation strings from evidence items
    valid_citations = set()
    
    if evidence_items:
        # Use evidence items to build citation format -> doc_id mapping
        for item in evidence_items:
            # Extract date from timestamp (YYYY-MM-DD format)
            timestamp_date = item.timestamp[:10] if len(item.timestamp) >= 10 else item.timestamp
            citation_format = f"[DOC:{item.doc_type}:{item.stock_code}:{timestamp_date}]"
            valid_citations.add(citation_format)
    
    # Also build citation patterns from doc_ids
    for doc_id in allowed_doc_ids:
        # Parse doc_id: format is typically "type:key:timestamp" or "type:stock:datetime"
        parts = doc_id.split(':', 2)
        if len(parts) >= 3:
            doc_type = parts[0]
            # For trends: "trends:stock:datetime"
            # For news: "news:id:timestamp" -> need to get stock_code from evidence
            # For trade_history: "trade_history:stock:timestamp"
            
            # Try to extract stock_code and date
            if doc_type == "trends":
                stock_code = parts[1]
                timestamp = parts[2]
                date = timestamp[:10] if len(timestamp) >= 10 else timestamp
                citation_format = f"[DOC:{doc_type}:{stock_code}:{date}]"
                valid_citations.add(citation_format)
            elif doc_type == "trade_history":
                stock_code = parts[1]
                timestamp = parts[2]
                date = timestamp[:10] if len(timestamp) >= 10 else timestamp
                citation_format = f"[DOC:{doc_type}:{stock_code}:{date}]"
                valid_citations.add(citation_format)
            # For news, doc_id format is "news:id:timestamp", so we need evidence_items
    
    # Replace citations that are not in valid_citations
    def repl(m):
        citation_str = m.group(0)  # Full citation: [DOC:type:stock:date]
        if citation_str in valid_citations:
            return citation_str  # Keep allowed citation
        else:
            logger.warning(f"Removing invalid citation: {citation_str} (not in evidence pack)")
            return ''  # Remove disallowed citation
    
    cleaned = CITATION_PATTERN.sub(repl, text)
    
    # Collapse repeated whitespace from removals
    cleaned = re.sub(r'[ \t]+', ' ', cleaned)
    cleaned = re.sub(r'\n{3,}', '\n\n', cleaned)
    
    return cleaned.strip()

