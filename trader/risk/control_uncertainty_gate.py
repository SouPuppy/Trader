"""
Uncertainty Gate 风险管理器
基于 LLM 不确定性评估的风险控制机制

当 LLM 对自身判断越不确定，系统应当越保守：
- 不确定性低 → 正常执行
- 不确定性中 → 降低仓位
- 不确定性高 → 拒绝交易
"""
from typing import Tuple, Optional
from trader.risk.RiskManager import RiskManager, RiskContext
from trader.risk.OrderIntent import OrderIntent
from trader.logger import get_logger

logger = get_logger(__name__)


class UncertaintyGateRiskManager(RiskManager):
    """
    不确定性门控风险管理器
    
    基于 OrderIntent 中的 confidence 字段进行不确定性评估：
    - confidence >= high_threshold: 低不确定性，正常执行
    - medium_threshold <= confidence < high_threshold: 中等不确定性，降低仓位
    - confidence < medium_threshold: 高不确定性，拒绝交易
    """
    
    def __init__(self, name: str = "UncertaintyGate",
                 high_threshold: float = 0.7,
                 medium_threshold: float = 0.4,
                 scale_down_factor: float = 0.5):
        """
        初始化不确定性门控风险管理器
        
        Args:
            name: 风险管理器名称
            high_threshold: 高置信度阈值，confidence >= high_threshold 时正常执行
            medium_threshold: 中等置信度阈值，confidence < medium_threshold 时拒绝交易
            scale_down_factor: 中等不确定性时的仓位缩放因子（0-1）
        """
        super().__init__(name)
        
        if not (0.0 <= medium_threshold <= high_threshold <= 1.0):
            raise ValueError(
                f"阈值必须满足 0 <= medium_threshold <= high_threshold <= 1.0, "
                f"got medium={medium_threshold}, high={high_threshold}"
            )
        
        if not (0.0 <= scale_down_factor <= 1.0):
            raise ValueError(f"scale_down_factor 必须在 [0.0, 1.0] 范围内, got {scale_down_factor}")
        
        self.high_threshold = high_threshold
        self.medium_threshold = medium_threshold
        self.scale_down_factor = scale_down_factor
    
    def validate(self, ctx: RiskContext, order: OrderIntent) -> Tuple[bool, Optional[str]]:
        """
        验证订单是否可以通过（基于不确定性评估）
        
        Args:
            ctx: 风险控制上下文
            order: 订单意图
            
        Returns:
            Tuple[bool, Optional[str]]: (是否通过, 拒绝原因)
        """
        # 如果订单没有 confidence 字段，默认通过（向后兼容）
        if order.confidence is None:
            logger.debug(f"[{self.name}] 订单 {order.symbol} 没有 confidence 字段，默认通过")
            return True, None
        
        # 高不确定性：拒绝交易
        if order.confidence < self.medium_threshold:
            reason = (
                f"不确定性过高 (confidence={order.confidence:.2f} < {self.medium_threshold}), "
                f"拒绝交易"
            )
            logger.info(f"[{self.name}] {reason}: {order}")
            return False, reason
        
        # 低或中等不确定性：通过验证（中等不确定性会在 adjust 中处理）
        return True, None
    
    def adjust(self, ctx: RiskContext, order: OrderIntent) -> OrderIntent:
        """
        调整订单（基于不确定性评估）
        
        中等不确定性时，降低仓位规模
        
        Args:
            ctx: 风险控制上下文
            order: 订单意图
            
        Returns:
            OrderIntent: 调整后的订单意图
        """
        # 如果订单没有 confidence 字段，不调整（向后兼容）
        if order.confidence is None:
            return order
        
        # 中等不确定性：降低仓位
        if self.medium_threshold <= order.confidence < self.high_threshold:
            logger.info(
                f"[{self.name}] 中等不确定性 (confidence={order.confidence:.2f}), "
                f"降低仓位: {order.symbol}, 缩放因子={self.scale_down_factor}"
            )
            
            # 创建调整后的订单
            adjusted_order = OrderIntent(
                symbol=order.symbol,
                side=order.side,
                timestamp=order.timestamp,
                qty=order.qty,
                target_weight=(
                    order.target_weight * self.scale_down_factor 
                    if order.target_weight is not None 
                    else None
                ),
                price_type=order.price_type,
                limit_price=order.limit_price,
                agent_name=order.agent_name,
                prompt_version=order.prompt_version,
                confidence=order.confidence,
                reason_tags=order.reason_tags,
                metadata={
                    **order.metadata,
                    "uncertainty_adjusted": True,
                    "original_weight": order.target_weight,
                    "scale_factor": self.scale_down_factor
                }
            )
            
            # 如果使用 qty，也进行缩放
            if adjusted_order.qty is not None:
                adjusted_order.qty = int(adjusted_order.qty * self.scale_down_factor)
                if adjusted_order.qty == 0:
                    adjusted_order.qty = 1  # 至少保留 1 股
            
            return adjusted_order
        
        # 低不确定性：不调整
        return order
    
    def __repr__(self):
        return (
            f"{self.__class__.__name__}(name='{self.name}', "
            f"high_threshold={self.high_threshold}, "
            f"medium_threshold={self.medium_threshold}, "
            f"scale_down_factor={self.scale_down_factor})"
        )

