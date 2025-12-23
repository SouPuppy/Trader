"""
订单意图/订单结构定义
统一的订单表示，用于 Agent → RiskManager → Engine 的流程
"""
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, Dict, Any
from enum import Enum


class OrderSide(Enum):
    """订单方向"""
    BUY = "BUY"
    SELL = "SELL"


class PriceType(Enum):
    """价格类型"""
    MKT = "MKT"  # 市价单
    LMT = "LMT"  # 限价单


@dataclass
class OrderIntent:
    """
    订单意图/订单结构
    
    统一的订单表示，包含：
    - 基本信息：symbol, side, qty/target_weight, price_type, timestamp
    - 元数据：生成来源（agent_name, prompt_version）
    - 可选信息：confidence, reason_tags（用于风控加权）
    """
    symbol: str
    side: OrderSide
    timestamp: str  # 交易日，格式: YYYY-MM-DD
    
    # 数量或目标权重（二选一）
    qty: Optional[int] = None  # 股数（用于 SELL 或精确买入）
    target_weight: Optional[float] = None  # 目标权重（0-1，用于 BUY）
    
    # 价格类型
    price_type: PriceType = PriceType.MKT
    
    # 生成来源
    agent_name: str = "unknown"
    prompt_version: Optional[str] = None
    
    # 可选：置信度和理由标签
    confidence: Optional[float] = None  # 0-1，用于风控加权
    reason_tags: Optional[list] = field(default_factory=list)  # 理由标签列表
    
    # 限价单价格（仅当 price_type=LMT 时使用）
    limit_price: Optional[float] = None
    
    # 元数据（用于存储额外信息）
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """验证订单意图的有效性"""
        if self.side == OrderSide.BUY:
            if self.qty is None and self.target_weight is None:
                raise ValueError("BUY 订单必须指定 qty 或 target_weight")
            if self.qty is not None and self.target_weight is not None:
                raise ValueError("BUY 订单不能同时指定 qty 和 target_weight")
        elif self.side == OrderSide.SELL:
            if self.qty is None:
                raise ValueError("SELL 订单必须指定 qty")
            if self.target_weight is not None:
                raise ValueError("SELL 订单不能指定 target_weight")
        
        if self.price_type == PriceType.LMT and self.limit_price is None:
            raise ValueError("限价单必须指定 limit_price")
        
        if self.confidence is not None and not (0.0 <= self.confidence <= 1.0):
            raise ValueError("confidence 必须在 [0.0, 1.0] 范围内")
    
    def __repr__(self):
        """字符串表示"""
        parts = [f"OrderIntent({self.side.value} {self.symbol}"]
        
        if self.qty is not None:
            parts.append(f"qty={self.qty}")
        if self.target_weight is not None:
            parts.append(f"target_weight={self.target_weight:.2%}")
        
        parts.append(f"price_type={self.price_type.value}")
        parts.append(f"date={self.timestamp}")
        parts.append(f"agent={self.agent_name}")
        
        if self.confidence is not None:
            parts.append(f"confidence={self.confidence:.2f}")
        
        parts.append(")")
        return ", ".join(parts)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "symbol": self.symbol,
            "side": self.side.value,
            "timestamp": self.timestamp,
            "qty": self.qty,
            "target_weight": self.target_weight,
            "price_type": self.price_type.value,
            "agent_name": self.agent_name,
            "prompt_version": self.prompt_version,
            "confidence": self.confidence,
            "reason_tags": self.reason_tags,
            "limit_price": self.limit_price,
            "metadata": self.metadata,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "OrderIntent":
        """从字典创建"""
        data = data.copy()
        data["side"] = OrderSide(data["side"])
        data["price_type"] = PriceType(data["price_type"])
        return cls(**data)

