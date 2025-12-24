"""
LLM 输出格式定义
RAG的终点不是"生成一段话"，而是生成一个可执行、可统计的结构化结果
"""
from dataclasses import dataclass
from typing import List, Optional, Dict, Any
from enum import Enum


class Decision(str, Enum):
    """交易决策"""
    TRADE = "TRADE"
    NO_TRADE = "NO_TRADE"


class RiskFlag(str, Enum):
    """风险标志"""
    HIGH_VOLATILITY = "HIGH_VOLATILITY"
    LOW_CONFIDENCE = "LOW_CONFIDENCE"
    INSUFFICIENT_EVIDENCE = "INSUFFICIENT_EVIDENCE"
    CONFLICTING_SIGNALS = "CONFLICTING_SIGNALS"
    MARKET_CLOSED = "MARKET_CLOSED"
    POSITION_LIMIT = "POSITION_LIMIT"


@dataclass
class RAGOutput:
    """
    LLM输出格式（结构化、可引用）
    
    核心原则：
    - 没有evidence_ids就不能因为新闻做判断（否则就是幻觉）
    - 所有决策必须可追溯、可复现
    """
    decision: Decision  # 决策：TRADE | NO_TRADE
    confidence: float  # 置信度：0~1
    reasons: List[str]  # 原因（短条目）
    evidence_ids: List[str]  # 证据ID列表（必须引用doc_id）
    risk_flags: List[RiskFlag]  # 风险标志列表
    metadata: Dict[str, Any] = None  # 额外元数据
    
    def __post_init__(self):
        """验证输出格式"""
        if self.confidence < 0 or self.confidence > 1:
            raise ValueError(f"confidence必须在[0, 1]范围内，当前值: {self.confidence}")
        
        if not self.evidence_ids:
            # 如果没有证据，应该设置INSUFFICIENT_EVIDENCE标志
            if RiskFlag.INSUFFICIENT_EVIDENCE not in self.risk_flags:
                self.risk_flags.append(RiskFlag.INSUFFICIENT_EVIDENCE)
        
        if self.metadata is None:
            self.metadata = {}
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'decision': self.decision.value,
            'confidence': self.confidence,
            'reasons': self.reasons,
            'evidence_ids': self.evidence_ids,
            'risk_flags': [flag.value for flag in self.risk_flags],
            'metadata': self.metadata,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'RAGOutput':
        """从字典创建RAGOutput"""
        return cls(
            decision=Decision(data['decision']),
            confidence=data['confidence'],
            reasons=data['reasons'],
            evidence_ids=data['evidence_ids'],
            risk_flags=[RiskFlag(flag) for flag in data['risk_flags']],
            metadata=data.get('metadata', {}),
        )
    
    def is_valid(self) -> bool:
        """
        验证输出是否有效
        
        规则：
        - 如果有TRADE决策，必须有evidence_ids
        - 如果有INSUFFICIENT_EVIDENCE标志，confidence应该较低
        """
        if self.decision == Decision.TRADE:
            if not self.evidence_ids:
                return False
        
        if RiskFlag.INSUFFICIENT_EVIDENCE in self.risk_flags:
            if self.confidence > 0.5:
                # 如果证据不足但置信度高，可能有问题
                return False
        
        return True
    
    def get_summary(self) -> str:
        """获取输出摘要（用于日志）"""
        return (
            f"Decision: {self.decision.value}, "
            f"Confidence: {self.confidence:.2f}, "
            f"Evidence: {len(self.evidence_ids)} docs, "
            f"Risks: {[f.value for f in self.risk_flags]}"
        )

