"""
Prompt Builder
Generate strictly formatted prompt to ensure LLM only uses evidence
"""
from typing import List
import sys
from pathlib import Path
import json

project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from trader.rag.types import RagRequest, EvidencePack
from trader.rag.calculate_trends import calculate_trends_statistics, format_trends_summary
from trader.rag import get_rag_logger

logger = get_rag_logger(__name__)


def build_prompt(request: RagRequest, evidence: EvidencePack, degradation_msg: str = None) -> str:
    """
    Build prompt
    
    Args:
        request: RAG request
        evidence: Evidence pack
        
    Returns:
        prompt string
    """
    # System prompt
    system_prompt = """You are a professional financial analyst assistant. Your task is to answer user questions based on the provided evidence.

**CRITICAL CITATION FORMAT RULES:**
1. **ONLY** use this exact citation format: [DOC:type:stock:date]
   - Example: [DOC:trends:AAPL.O:2023-12-28]
   - Example: [DOC:news_piece:NVDA.O:2023-12-15]
   - Example: [DOC:trade_history:TSLA.O:2023-12-10]
2. **STRICTLY FORBIDDEN:**
   - NO comma-separated citations: [DOC:A, DOC:B] ❌
   - NO range citations: [DOC:A to DOC:B] or [DOC:A through DOC:B] ❌
   - NO truncated IDs: [DOC:trends:AAPL] ❌
   - NO doc_id: prefix: doc_id: trends:AAPL.O:2023-12-28 ❌
3. **Multiple citations:** Use separate brackets: [DOC:trends:AAPL.O:2023-12-28] [DOC:trends:AAPL.O:2023-12-27]
4. **Citation format:** [DOC:doc_type:stock_code:timestamp]
   - doc_type: trends | news_piece | trade_history
   - stock_code: e.g., AAPL.O, NVDA.O
   - timestamp: YYYY-MM-DD format (date only, no time)

**Important Constraints:**
1. You can only use information from the provided evidence (EvidencePack) to answer questions
2. All references MUST use the [DOC:type:stock:date] format - NO EXCEPTIONS
3. If evidence is insufficient, you must clearly state "Insufficient evidence, cannot answer"
4. Do not fabricate or speculate information that does not exist in the evidence
5. If there are contradictions in the evidence, point them out and explain
6. **IMPORTANT: You must answer in English only. All responses must be in English.**

**Output Format Requirements:**
- First provide a brief answer (1-2 sentences)
- Then list key evidence supporting the answer (with [DOC:type:stock:date] citations)
- Finally explain the limitations of the evidence (if any)"""

    # Format evidence pack
    evidence_json = _format_evidence_pack(evidence)
    
    # Build allowed citation list
    allowed_citations = []
    for item in evidence.items:
        timestamp_date = item.timestamp[:10] if len(item.timestamp) >= 10 else item.timestamp
        citation_format = f"[DOC:{item.doc_type}:{item.stock_code}:{timestamp_date}]"
        allowed_citations.append(citation_format)
    
    allowed_citations_str = "\n".join(f"- {cit}" for cit in allowed_citations) if allowed_citations else "**NONE - DO NOT USE CITATIONS**"
    
    # Calculate trends statistics for market_state questions (reduce LLM hallucination)
    trends_summary = ""
    if evidence.task_type == "market_state":
        trends_stats = calculate_trends_statistics(evidence)
        if trends_stats:
            trends_summary = f"\n\n**Calculated Trend Statistics (use these exact values, do not invent numbers):**\n{format_trends_summary(trends_stats)}"
    
    # User prompt
    degradation_note = ""
    if degradation_msg:
        degradation_note = f"\n\n**IMPORTANT:** {degradation_msg}\nPlease acknowledge this limitation in your answer."
    
    citation_constraint = ""
    if not allowed_citations:
        citation_constraint = "\n\n**CRITICAL: ALLOWED_DOC_IDS is EMPTY. You MUST NOT use any citations. State 'Insufficient evidence, cannot answer' if you cannot answer without citations."
    else:
        citation_constraint = f"\n\n**ALLOWED CITATIONS (you can ONLY use these):**\n{allowed_citations_str}\n\n**If ALLOWED_DOC_IDS is empty or a citation is not in this list, do NOT use it.**"
    
    user_prompt = f"""**Question:**
{request.question}

**Decision Time:** {request.decision_time}
**Stock Code:** {request.stock_code or "Not specified"}
{degradation_note}{trends_summary}{citation_constraint}

**Evidence Pack (EvidencePack):**
```json
{evidence_json}
```

Please answer the question based on the above evidence in English. If evidence is insufficient, please clearly state so.
**For market_state questions:** Use the calculated trend statistics above for numerical values. Do not invent or estimate numbers."""

    # Combine full prompt
    full_prompt = f"""<system>
{system_prompt}
</system>

<user>
{user_prompt}
</user>"""

    return full_prompt


def _format_evidence_pack(evidence: EvidencePack) -> str:
    """Format evidence pack as JSON with citation format examples"""
    items_list = []
    for item in evidence.items:
        # Extract date from timestamp (YYYY-MM-DD format)
        timestamp_date = item.timestamp[:10] if len(item.timestamp) >= 10 else item.timestamp
        
        # Build citation format: [DOC:type:stock:date]
        citation_format = f"[DOC:{item.doc_type}:{item.stock_code}:{timestamp_date}]"
        
        item_dict = {
            "doc_id": item.doc_id,
            "citation_format": citation_format,  # Add citation format example
            "doc_type": item.doc_type,
            "stock_code": item.stock_code,
            "timestamp": item.timestamp,
            "key_facts": item.key_facts,
            "signals": item.signals,
            "relevance_score": item.relevance_score
        }
        items_list.append(item_dict)
    
    evidence_dict = {
        "task_type": evidence.task_type,
        "stock_code": evidence.stock_code,
        "decision_time": evidence.decision_time,
        "items": items_list
    }
    
    return json.dumps(evidence_dict, indent=2, ensure_ascii=False)
