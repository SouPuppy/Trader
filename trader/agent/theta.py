"""
参数θ管理：层级式多资产交易系统的可调参数
"""
from dataclasses import dataclass, field
from typing import Dict, Optional, Literal
from datetime import datetime
import json


@dataclass
class Theta:
    """
    层级式多资产交易系统的参数集合
    
    参数范围：
    - gross_exposure: 总仓位上限 [0.0, 1.0]
    - max_w: 单票上限 [0.01, 0.20]
    - turnover_cap: 日/周换手上限 [0.0, 0.50]
    - risk_mode: 风险档位 {risk_on, neutral, risk_off}
    - enter_th: 进场阈值（作用于score或概率）
    - exit_th: 出场阈值（作用于score或概率）
    - factor_weights_delta: 对news/trend/value权重的微调（可选）
    """
    
    # 仓位约束参数
    gross_exposure: float = 1.0  # 总仓位上限 [0.0, 1.0]
    max_w: float = 0.20  # 单票上限 [0.01, 0.20]
    turnover_cap: float = 0.50  # 日/周换手上限 [0.0, 0.50]
    
    # 风险模式
    risk_mode: Literal["risk_on", "neutral", "risk_off"] = "neutral"
    
    # 进出场阈值
    enter_th: float = 0.0  # 进场阈值
    exit_th: float = -0.1  # 出场阈值
    
    # 因子权重微调（可选）
    factor_weights_delta: Optional[Dict[str, float]] = field(default_factory=dict)
    # 例如: {"w_T": 0.0, "w_V": 0.0, "w_F": 0.0, "w_N": 0.0}
    
    # 元数据
    timestamp: Optional[datetime] = None
    reflection_id: Optional[int] = None  # 反思轮次ID
    
    def __post_init__(self):
        """参数校验和裁剪"""
        # 裁剪gross_exposure到[0.0, 1.0]
        self.gross_exposure = max(0.0, min(1.0, self.gross_exposure))
        
        # 裁剪max_w到[0.01, 0.20]
        self.max_w = max(0.01, min(0.20, self.max_w))
        
        # 裁剪turnover_cap到[0.0, 0.50]
        self.turnover_cap = max(0.0, min(0.50, self.turnover_cap))
        
        # 确保risk_mode是有效值
        if self.risk_mode not in ["risk_on", "neutral", "risk_off"]:
            self.risk_mode = "neutral"
        
        # 如果没有timestamp，设置为当前时间
        if self.timestamp is None:
            self.timestamp = datetime.now()
    
    def to_dict(self) -> Dict:
        """转换为字典"""
        return {
            "gross_exposure": self.gross_exposure,
            "max_w": self.max_w,
            "turnover_cap": self.turnover_cap,
            "risk_mode": self.risk_mode,
            "enter_th": self.enter_th,
            "exit_th": self.exit_th,
            "factor_weights_delta": self.factor_weights_delta.copy() if self.factor_weights_delta else {},
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
            "reflection_id": self.reflection_id
        }
    
    def to_json(self) -> str:
        """转换为JSON字符串"""
        return json.dumps(self.to_dict(), indent=2, ensure_ascii=False)
    
    @classmethod
    def from_dict(cls, data: Dict) -> "Theta":
        """从字典创建"""
        # 处理timestamp
        if "timestamp" in data and data["timestamp"]:
            if isinstance(data["timestamp"], str):
                data["timestamp"] = datetime.fromisoformat(data["timestamp"])
        
        return cls(**data)
    
    @classmethod
    def from_json(cls, json_str: str) -> "Theta":
        """从JSON字符串创建"""
        data = json.loads(json_str)
        return cls.from_dict(data)
    
    def get_risk_scale(self) -> float:
        """
        根据risk_mode返回缩放因子
        
        Returns:
            float: 缩放因子
                - risk_on: 1.0 (不缩放)
                - neutral: 0.8 (降低20%)
                - risk_off: 0.5 (降低50%)
        """
        scale_map = {
            "risk_on": 1.0,
            "neutral": 0.8,
            "risk_off": 0.5
        }
        return scale_map.get(self.risk_mode, 0.8)
    
    def copy(self) -> "Theta":
        """创建副本"""
        return Theta(
            gross_exposure=self.gross_exposure,
            max_w=self.max_w,
            turnover_cap=self.turnover_cap,
            risk_mode=self.risk_mode,
            enter_th=self.enter_th,
            exit_th=self.exit_th,
            factor_weights_delta=self.factor_weights_delta.copy() if self.factor_weights_delta else {},
            timestamp=self.timestamp,
            reflection_id=self.reflection_id
        )
    
    def __repr__(self):
        return (
            f"Theta(gross_exposure={self.gross_exposure:.2f}, "
            f"max_w={self.max_w:.2f}, turnover_cap={self.turnover_cap:.2f}, "
            f"risk_mode={self.risk_mode}, enter_th={self.enter_th:.3f}, "
            f"exit_th={self.exit_th:.3f})"
        )

