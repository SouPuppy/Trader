"""
Evidence Gate
Check if required evidence exists, degrade if missing
"""
from typing import Optional, Dict, List, Tuple
from datetime import datetime, timedelta
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from trader.rag.types import EvidencePack, TaskType, RetrievalPlan
from trader.rag.db.queries import get_db_connection
from trader.rag import get_rag_logger

logger = get_rag_logger(__name__)


def check_data_coverage(plan: RetrievalPlan) -> Optional[str]:
    """
    Check data coverage before retrieval (preflight check)
    
    Args:
        plan: Retrieval plan
        
    Returns:
        Degradation message if coverage is insufficient, None otherwise
    """
    conn = get_db_connection()
    cursor = conn.cursor()
    
    try:
        # Check trends coverage
        trends_need = plan.needs.get("trends")
        if trends_need and trends_need.enable:
            cursor.execute("""
                SELECT COUNT(*) as n, MIN(datetime) as min_d, MAX(datetime) as max_d
                FROM features
                WHERE stock_code = ? AND datetime BETWEEN ? AND ?
            """, (plan.stock_code, plan.time_start[:10], plan.time_end[:10]))
            
            row = cursor.fetchone()
            if row:
                count = row[0] or 0
                min_date = row[1]
                max_date = row[2]
                
                # Calculate expected days
                try:
                    start_dt = datetime.fromisoformat(plan.time_start[:10])
                    end_dt = datetime.fromisoformat(plan.time_end[:10])
                    expected_days = (end_dt - start_dt).days + 1
                    
                    # Check coverage ratio
                    if expected_days > 0:
                        coverage_ratio = count / expected_days
                        
                        # If coverage is very low (< 30%), warn
                        if coverage_ratio < 0.3:
                            return (f"Insufficient trend data coverage: only {count} data point(s) found "
                                   f"for {expected_days} expected days ({coverage_ratio:.1%} coverage). "
                                   f"Date range: {min_date or 'N/A'} to {max_date or 'N/A'}. "
                                   "The analysis may have significant gaps.")
                except (ValueError, TypeError):
                    pass
        
        # Check news coverage
        news_need = plan.needs.get("news_piece")
        if news_need and news_need.enable:
            cursor.execute("""
                SELECT COUNT(*) as n, MIN(datetime) as min_d, MAX(datetime) as max_d
                FROM raw_data
                WHERE stock_code = ? 
                  AND datetime BETWEEN ? AND ?
                  AND news IS NOT NULL AND news != ''
            """, (plan.stock_code, plan.time_start[:10], plan.time_end[:10]))
            
            row = cursor.fetchone()
            if row:
                count = row[0] or 0
                min_date = row[1]
                max_date = row[2]
                
                # For news, if count is 0 and task requires news, warn
                if count == 0 and plan.task_type == "news_impact":
                    return (f"No news data found for the specified time window. "
                           f"Date range requested: {plan.time_start[:10]} to {plan.time_end[:10]}. "
                           "Cannot perform news impact analysis without news data.")
        
        # Check trade_history coverage
        trade_need = plan.needs.get("trade_history")
        if trade_need and trade_need.enable:
            cursor.execute("""
                SELECT COUNT(*) as n, MIN(trade_time) as min_t, MAX(trade_time) as max_t
                FROM trade_history
                WHERE stock_code = ? AND trade_time BETWEEN ? AND ?
            """, (plan.stock_code, plan.time_start, plan.time_end))
            
            row = cursor.fetchone()
            if row:
                count = row[0] or 0
                
                # If task requires trade history but none found, warn
                if count == 0 and plan.task_type == "trade_explain":
                    return (f"No trading history found for the specified time window. "
                           f"Date range requested: {plan.time_start[:10]} to {plan.time_end[:10]}. "
                           "Cannot explain trading decisions without trade history data.")
    
    except Exception as e:
        logger.warning(f"Data coverage check failed: {e}", exc_info=True)
    finally:
        conn.close()
    
    return None


def check_evidence_gate(evidence: EvidencePack) -> Optional[str]:
    """
    Check if required evidence exists, return degradation message if missing
    
    Args:
        evidence: Evidence pack
        
    Returns:
        Degradation message if evidence is insufficient, None otherwise
    """
    task_type = evidence.task_type
    
    # Count evidence by type
    evidence_by_type = {}
    for item in evidence.items:
        doc_type = item.doc_type
        evidence_by_type[doc_type] = evidence_by_type.get(doc_type, 0) + 1
    
    # Task-specific evidence gates
    if task_type == "trade_explain":
        # trade_explain MUST have trade_history
        trade_history_count = evidence_by_type.get("trade_history", 0)
        if trade_history_count == 0:
            # Check if we have trends as background (but can't use as trade history)
            trends_count = evidence_by_type.get("trends", 0)
            if trends_count > 0:
                return ("No trading history records found for the specified time window. "
                       "I can only provide market background from trend features, but cannot "
                       "explain specific trading decisions without trade history data.")
            else:
                return ("No trading history records found for the specified time window. "
                       "Cannot answer questions about trading history without trade data.")
    
    elif task_type == "news_impact":
        # news_impact needs at least 2 valid news items
        news_count = evidence_by_type.get("news_piece", 0)
        min_news_threshold = 2
        if news_count < min_news_threshold:
            trends_count = evidence_by_type.get("trends", 0)
            if trends_count > 0:
                return (f"Only {news_count} news item(s) found (minimum {min_news_threshold} required for news impact analysis). "
                       "I can only provide market state information from trend features, but cannot "
                       "perform news attribution analysis without sufficient news data.")
            else:
                return (f"Insufficient news data: only {news_count} news item(s) found "
                       f"(minimum {min_news_threshold} required for news impact analysis).")
    
    elif task_type == "market_state":
        # market_state needs trends coverage
        trends_count = evidence_by_type.get("trends", 0)
        if trends_count == 0:
            return ("No trend feature data found for the specified time window. "
                   "Cannot provide market state analysis without trend data.")
        
        # Check coverage for "last N days" queries
        # This is a simplified check - in production, you'd calculate actual coverage
        # For now, if trends_count is very low, warn about coverage gaps
        if trends_count < 5:
            return (f"Limited trend data coverage: only {trends_count} data point(s) found. "
                   "The analysis may have gaps in the specified time window.")
    
    # All checks passed
    return None


def apply_evidence_gate(evidence: EvidencePack) -> Tuple[EvidencePack, Optional[str]]:
    """
    Apply evidence gate and return degraded evidence pack if needed
    
    Args:
        evidence: Original evidence pack
        
    Returns:
        Tuple of (evidence_pack, degradation_message)
        - If gate passes: (original_evidence, None)
        - If gate fails: (degraded_evidence, degradation_message)
    """
    degradation_msg = check_evidence_gate(evidence)
    
    if degradation_msg:
        logger.warning(f"Evidence gate triggered for task_type={evidence.task_type}: {degradation_msg}")
        # Return evidence with degradation message (will be added to answer)
        return evidence, degradation_msg
    
    return evidence, None

