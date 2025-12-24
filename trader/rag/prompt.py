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
from trader.logger import get_logger

logger = get_logger(__name__)


def build_prompt(request: RagRequest, evidence: EvidencePack) -> str:
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

**Important Constraints:**
1. You can only use information from the provided evidence (EvidencePack) to answer questions
2. All references must clearly indicate doc_id (format: doc_id)
3. If evidence is insufficient, you must clearly state "Insufficient evidence, cannot answer"
4. Do not fabricate or speculate information that does not exist in the evidence
5. If there are contradictions in the evidence, point them out and explain
6. **IMPORTANT: You must answer in English only. All responses must be in English.**

**Output Format Requirements:**
- First provide a brief answer (1-2 sentences)
- Then list key evidence supporting the answer (with doc_id references)
- Finally explain the limitations of the evidence (if any)"""

    # Format evidence pack
    evidence_json = _format_evidence_pack(evidence)
    
    # User prompt
    user_prompt = f"""**Question:**
{request.question}

**Decision Time:** {request.decision_time}
**Stock Code:** {request.stock_code or "Not specified"}

**Evidence Pack (EvidencePack):**
```json
{evidence_json}
```

Please answer the question based on the above evidence in English. If evidence is insufficient, please clearly state so."""

    # Combine full prompt
    full_prompt = f"""<system>
{system_prompt}
</system>

<user>
{user_prompt}
</user>"""

    return full_prompt


def _format_evidence_pack(evidence: EvidencePack) -> str:
    """Format evidence pack as JSON"""
    evidence_dict = {
        "task_type": evidence.task_type,
        "stock_code": evidence.stock_code,
        "decision_time": evidence.decision_time,
        "items": [
            {
                "doc_id": item.doc_id,
                "doc_type": item.doc_type,
                "stock_code": item.stock_code,
                "timestamp": item.timestamp,
                "key_facts": item.key_facts,
                "signals": item.signals,
                "relevance_score": item.relevance_score
            }
            for item in evidence.items
        ]
    }
    
    return json.dumps(evidence_dict, indent=2, ensure_ascii=False)
