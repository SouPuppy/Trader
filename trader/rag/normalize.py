"""
Request Normalizer
Parse user input into unified request
"""
import re
from datetime import datetime, timedelta
from typing import Optional
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from trader.rag.types import RagRequest
from trader.logger import get_logger

logger = get_logger(__name__)


def normalize_request(
    question: str,
    stock_code: Optional[str] = None,
    decision_time: Optional[str] = None,
    frequency: str = "1d",
    context: Optional[dict] = None
) -> RagRequest:
    """
    Normalize RAG request
    
    Args:
        question: User question
        stock_code: Stock code (optional, can be extracted from question)
        decision_time: Decision time (ISO8601, REQUIRED - must be provided explicitly to simulate historical queries)
        frequency: Frequency ('1d'/'1h'...)
        context: Context (optional: current time, default stock_code, user selected symbols, etc.)
        
    Returns:
        RagRequest
        
    Raises:
        ValueError: If decision_time is not provided
    """
    # Get default values from context
    if context:
        stock_code = stock_code or context.get('stock_code')
        decision_time = decision_time or context.get('decision_time')
        frequency = frequency or context.get('frequency', '1d')
    
    # Extract stock_code from question (if not provided)
    if not stock_code:
        # Try to extract stock code from question (e.g., AAPL.O, TSLA.O, etc.)
        pattern = r'\b([A-Z]{1,5}\.[A-Z]{1,3})\b'
        matches = re.findall(pattern, question.upper())
        if matches:
            stock_code = matches[0]
            logger.debug(f"Extracted stock code from question: {stock_code}")
    
    # Decision time is REQUIRED - must be provided explicitly to simulate historical queries
    if not decision_time:
        raise ValueError(
            "decision_time is required and must be provided explicitly. "
            "This ensures queries are simulated at specific historical time points. "
            "Example: decision_time='2023-12-15T00:00:00'"
        )
    
    # Ensure decision_time is ISO8601 format
    try:
        # Try to parse and reformat
        dt = datetime.fromisoformat(decision_time.replace('Z', '+00:00'))
        decision_time = dt.isoformat()
    except (ValueError, AttributeError) as e:
        # If parsing fails, raise error instead of using current time
        raise ValueError(
            f"Cannot parse decision_time '{decision_time}'. "
            "Please provide a valid ISO8601 format datetime string. "
            "Example: '2023-12-15T00:00:00' or '2023-12-15T00:00:00Z'"
        ) from e
    
    return RagRequest(
        question=question.strip(),
        stock_code=stock_code,
        decision_time=decision_time,
        task_hint=None,
        frequency=frequency
    )

