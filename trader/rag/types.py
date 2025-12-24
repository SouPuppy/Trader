"""
RAG System Type Definitions
Unified data abstraction: Document / Candidate / Evidence
"""
from dataclasses import dataclass
from typing import Any, Dict, Optional, Literal, List

# Document types
DocType = Literal["trends", "trade_history", "news_piece"]

# Task types
TaskType = Literal["market_state", "trade_explain", "risk_check", "news_impact", "strategy_suggest"]


@dataclass(frozen=True)
class Document:
    """Unified document object"""
    doc_id: str                 # Globally unique (format: type:primary_key)
    doc_type: DocType
    stock_code: str
    timestamp: str              # ISO8601 (or SQLite TEXT datetime)
    payload: Dict[str, Any]     # Original fields (structured/semi-structured)
    text: Optional[str] = None  # Text for text retrieval/embedding (commonly used for news)
    embedding: Optional[List[float]] = None  # May not be here if using vector database


@dataclass(frozen=True)
class Candidate:
    """Retrieval candidate"""
    doc: Document
    recall_score: float                 # Recall stage score (BM25/similarity/rule score)
    recall_source: str                  # "bm25" | "vector" | "feature_knn" | "sql_filter"


@dataclass(frozen=True)
class EvidenceItem:
    """Final evidence card (only this for LLM)"""
    doc_id: str
    doc_type: DocType
    stock_code: str
    timestamp: str
    key_facts: List[str]                # 1-3 items
    signals: Dict[str, Any]             # Numerical signals (sentiment/impact/ret_5d/vol_z_20d etc.)
    relevance_score: float              # Final score after rerank


@dataclass(frozen=True)
class RagRequest:
    """RAG request"""
    question: str
    stock_code: Optional[str]
    decision_time: str       # ISO8601
    task_hint: Optional[str] = None  # Can be empty, Planner will determine later
    frequency: str = "1d"    # '1d'/'1h'...


@dataclass(frozen=True)
class RetrievalNeed:
    """Retrieval need configuration"""
    enable: bool
    recall_k: int           # Number of candidates returned in recall stage
    final_k: int            # Number of candidates retained finally


@dataclass(frozen=True)
class RetrievalPlan:
    """Retrieval plan"""
    task_type: TaskType
    stock_code: Optional[str]
    time_start: str         # ISO8601
    time_end: str           # ISO8601
    needs: Dict[DocType, RetrievalNeed]
    constraints: Dict[str, Any]   # impact threshold, max_trades, frequency, etc.
    semantic_queries: List[str]   # Semantic queries (for news)
    constraint_queries: List[str]  # Constraint queries (for SQL filtering)


@dataclass(frozen=True)
class EvidencePack:
    """Evidence pack"""
    task_type: str
    stock_code: Optional[str]
    decision_time: str
    items: List[EvidenceItem]


@dataclass(frozen=True)
class VerifiedAnswer:
    """Verified answer"""
    answer: str
    passed: bool
    violations: List[str]
    used_doc_ids: List[str]
    mode: str  # "normal" | "degraded"

