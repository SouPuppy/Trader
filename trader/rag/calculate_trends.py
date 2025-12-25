"""
Trends Calculator
Calculate market statistics from trends data to reduce LLM hallucination
"""
from typing import Dict, List, Optional
from datetime import datetime, timedelta
import statistics
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from trader.rag.types import EvidencePack, EvidenceItem
from trader.rag import get_rag_logger

logger = get_rag_logger(__name__)


def calculate_trends_statistics(evidence: EvidencePack) -> Dict[str, any]:
    """
    Calculate market statistics from trends evidence
    
    Args:
        evidence: Evidence pack containing trends items
        
    Returns:
        Dictionary with calculated statistics
    """
    trends_items = [item for item in evidence.items if item.doc_type == "trends"]
    
    if not trends_items:
        return {}
    
    stats = {
        "data_points": len(trends_items),
        "date_range": _get_date_range(trends_items),
        "returns": {},
        "volatility": {},
        "ratios": {}
    }
    
    # Extract numerical signals
    ret_1d_values = []
    ret_5d_values = []
    ret_20d_values = []
    vol_20d_values = []
    vol_60d_values = []
    vol_z_20d_values = []
    pe_ratios = []
    pb_ratios = []
    ps_ratios = []
    
    for item in trends_items:
        signals = item.signals or {}
        
        if "ret_1d" in signals and signals["ret_1d"] is not None:
            ret_1d_values.append(float(signals["ret_1d"]))
        if "ret_5d" in signals and signals["ret_5d"] is not None:
            ret_5d_values.append(float(signals["ret_5d"]))
        if "ret_20d" in signals and signals["ret_20d"] is not None:
            ret_20d_values.append(float(signals["ret_20d"]))
        if "vol_20d" in signals and signals["vol_20d"] is not None:
            vol_20d_values.append(float(signals["vol_20d"]))
        if "vol_60d" in signals and signals["vol_60d"] is not None:
            vol_60d_values.append(float(signals["vol_60d"]))
        if "vol_z_20d" in signals and signals["vol_z_20d"] is not None:
            vol_z_20d_values.append(float(signals["vol_z_20d"]))
        if "pe_ratio" in signals and signals["pe_ratio"] is not None:
            pe_ratios.append(float(signals["pe_ratio"]))
        if "pb_ratio" in signals and signals["pb_ratio"] is not None:
            pb_ratios.append(float(signals["pb_ratio"]))
        if "ps_ratio" in signals and signals["ps_ratio"] is not None:
            ps_ratios.append(float(signals["ps_ratio"]))
    
    # Calculate return statistics
    if ret_1d_values:
        stats["returns"]["ret_1d"] = {
            "mean": statistics.mean(ret_1d_values),
            "median": statistics.median(ret_1d_values),
            "min": min(ret_1d_values),
            "max": max(ret_1d_values),
            "std": statistics.stdev(ret_1d_values) if len(ret_1d_values) > 1 else 0.0,
            "positive_count": sum(1 for v in ret_1d_values if v > 0),
            "negative_count": sum(1 for v in ret_1d_values if v < 0),
            "count": len(ret_1d_values)
        }
    
    if ret_5d_values:
        stats["returns"]["ret_5d"] = {
            "mean": statistics.mean(ret_5d_values),
            "median": statistics.median(ret_5d_values),
            "min": min(ret_5d_values),
            "max": max(ret_5d_values),
            "std": statistics.stdev(ret_5d_values) if len(ret_5d_values) > 1 else 0.0,
            "count": len(ret_5d_values)
        }
    
    if ret_20d_values:
        stats["returns"]["ret_20d"] = {
            "mean": statistics.mean(ret_20d_values),
            "median": statistics.median(ret_20d_values),
            "min": min(ret_20d_values),
            "max": max(ret_20d_values),
            "std": statistics.stdev(ret_20d_values) if len(ret_20d_values) > 1 else 0.0,
            "count": len(ret_20d_values)
        }
    
    # Calculate volatility statistics
    if vol_20d_values:
        stats["volatility"]["vol_20d"] = {
            "mean": statistics.mean(vol_20d_values),
            "median": statistics.median(vol_20d_values),
            "min": min(vol_20d_values),
            "max": max(vol_20d_values),
            "count": len(vol_20d_values)
        }
    
    if vol_z_20d_values:
        stats["volatility"]["vol_z_20d"] = {
            "mean": statistics.mean(vol_z_20d_values),
            "median": statistics.median(vol_z_20d_values),
            "min": min(vol_z_20d_values),
            "max": max(vol_z_20d_values),
            "count": len(vol_z_20d_values),
            "above_2_count": sum(1 for v in vol_z_20d_values if v > 2.0),  # High volatility
            "below_neg2_count": sum(1 for v in vol_z_20d_values if v < -2.0)  # Low volume
        }
    
    # Calculate ratio statistics
    if pe_ratios:
        stats["ratios"]["pe_ratio"] = {
            "mean": statistics.mean(pe_ratios),
            "median": statistics.median(pe_ratios),
            "min": min(pe_ratios),
            "max": max(pe_ratios),
            "count": len(pe_ratios)
        }
    
    if pb_ratios:
        stats["ratios"]["pb_ratio"] = {
            "mean": statistics.mean(pb_ratios),
            "median": statistics.median(pb_ratios),
            "min": min(pb_ratios),
            "max": max(pb_ratios),
            "count": len(pb_ratios)
        }
    
    if ps_ratios:
        stats["ratios"]["ps_ratio"] = {
            "mean": statistics.mean(ps_ratios),
            "median": statistics.median(ps_ratios),
            "min": min(ps_ratios),
            "max": max(ps_ratios),
            "count": len(ps_ratios)
        }
    
    return stats


def _get_date_range(items: List[EvidenceItem]) -> Dict[str, str]:
    """Get date range from evidence items"""
    if not items:
        return {}
    
    dates = []
    for item in items:
        try:
            # Extract date part (YYYY-MM-DD)
            date_str = item.timestamp[:10] if len(item.timestamp) >= 10 else item.timestamp
            dates.append(date_str)
        except:
            continue
    
    if not dates:
        return {}
    
    dates.sort()
    return {
        "start": dates[0],
        "end": dates[-1],
        "days": len(set(dates))  # Unique days
    }


def format_trends_summary(stats: Dict) -> str:
    """
    Format trends statistics as a summary string for LLM
    
    Args:
        stats: Statistics dictionary from calculate_trends_statistics
        
    Returns:
        Formatted summary string
    """
    if not stats:
        return "No trend data available."
    
    lines = []
    lines.append(f"**Trend Data Coverage:** {stats.get('data_points', 0)} data points")
    
    date_range = stats.get("date_range", {})
    if date_range:
        lines.append(f"**Date Range:** {date_range.get('start', 'N/A')} to {date_range.get('end', 'N/A')} ({date_range.get('days', 0)} unique days)")
    
    returns = stats.get("returns", {})
    if returns:
        lines.append("\n**Return Statistics:**")
        for ret_type, ret_stats in returns.items():
            mean = ret_stats.get("mean", 0) * 100
            lines.append(f"  - {ret_type}: mean={mean:.2f}%, median={ret_stats.get('median', 0)*100:.2f}%, "
                        f"range=[{ret_stats.get('min', 0)*100:.2f}%, {ret_stats.get('max', 0)*100:.2f}%]")
            if "positive_count" in ret_stats:
                lines.append(f"    Positive days: {ret_stats['positive_count']}, Negative days: {ret_stats['negative_count']}")
    
    volatility = stats.get("volatility", {})
    if volatility:
        lines.append("\n**Volatility Statistics:**")
        for vol_type, vol_stats in volatility.items():
            lines.append(f"  - {vol_type}: mean={vol_stats.get('mean', 0):.4f}, "
                        f"range=[{vol_stats.get('min', 0):.4f}, {vol_stats.get('max', 0):.4f}]")
    
    ratios = stats.get("ratios", {})
    if ratios:
        lines.append("\n**Financial Ratios:**")
        for ratio_type, ratio_stats in ratios.items():
            lines.append(f"  - {ratio_type}: mean={ratio_stats.get('mean', 0):.2f}, "
                        f"range=[{ratio_stats.get('min', 0):.2f}, {ratio_stats.get('max', 0):.2f}]")
    
    return "\n".join(lines)

