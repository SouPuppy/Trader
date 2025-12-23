"""
Position Scaling 风险管理器
根据多种因素动态调整仓位大小

Position Scaling 策略：
- 基于账户权益：账户越大，单笔交易比例可能越小（风险分散）
- 基于波动率：波动率越高，仓位越小（Kelly Criterion 变种）
- 基于置信度：置信度越低，仓位越小（细粒度的不确定性控制）
- 基于历史表现：连续亏损后降低仓位，连续盈利后增加仓位
"""
from typing import Tuple, Optional, Dict, Any, List
from trader.risk.RiskManager import RiskManager, RiskContext
from trader.risk.OrderIntent import OrderIntent, OrderSide
from trader.logger import get_logger

logger = get_logger(__name__)


class PositionScalingRiskManager(RiskManager):
    """
    仓位缩放风险管理器
    
    根据多种因素动态调整仓位大小：
    1. 账户权益缩放：账户越大，单笔交易比例可能越小
    2. 波动率缩放：波动率越高，仓位越小
    3. 置信度缩放：置信度越低，仓位越小
    4. 历史表现缩放：连续亏损后降低仓位，连续盈利后增加仓位
    """
    
    def __init__(
        self,
        name: str = "PositionScalingRiskManager",
        # 账户权益缩放参数
        enable_equity_scaling: bool = True,
        base_equity: float = 1_000_000.0,  # 基准账户权益
        equity_scaling_factor: float = 0.5,  # 权益缩放因子（账户越大，比例越小）
        # 波动率缩放参数
        enable_volatility_scaling: bool = False,  # 默认关闭（需要额外的波动率数据）
        base_volatility: float = 0.02,  # 基准波动率（2%）
        volatility_scaling_factor: float = 1.0,  # 波动率缩放因子
        # 置信度缩放参数
        enable_confidence_scaling: bool = True,
        confidence_power: float = 1.0,  # 置信度幂次（1.0 = 线性，2.0 = 平方）
        # 历史表现缩放参数
        enable_performance_scaling: bool = True,
        consecutive_loss_threshold: int = 3,  # 连续亏损阈值
        consecutive_win_threshold: int = 3,  # 连续盈利阈值
        loss_scaling_factor: float = 0.5,  # 连续亏损后的缩放因子
        win_scaling_factor: float = 1.2,  # 连续盈利后的缩放因子（上限 1.0）
        max_scaling_factor: float = 1.0,  # 最大缩放因子（上限）
        min_scaling_factor: float = 0.1,  # 最小缩放因子（下限）
    ):
        """
        初始化仓位缩放风险管理器
        
        Args:
            name: 风险管理器名称
            enable_equity_scaling: 是否启用账户权益缩放
            base_equity: 基准账户权益
            equity_scaling_factor: 权益缩放因子（账户越大，比例越小）
            enable_volatility_scaling: 是否启用波动率缩放
            base_volatility: 基准波动率
            volatility_scaling_factor: 波动率缩放因子
            enable_confidence_scaling: 是否启用置信度缩放
            confidence_power: 置信度幂次（1.0 = 线性，2.0 = 平方）
            enable_performance_scaling: 是否启用历史表现缩放
            consecutive_loss_threshold: 连续亏损阈值
            consecutive_win_threshold: 连续盈利阈值
            loss_scaling_factor: 连续亏损后的缩放因子
            win_scaling_factor: 连续盈利后的缩放因子（上限 1.0）
            max_scaling_factor: 最大缩放因子（上限）
            min_scaling_factor: 最小缩放因子（下限）
        """
        super().__init__(name)
        
        self.enable_equity_scaling = enable_equity_scaling
        self.base_equity = base_equity
        self.equity_scaling_factor = equity_scaling_factor
        
        self.enable_volatility_scaling = enable_volatility_scaling
        self.base_volatility = base_volatility
        self.volatility_scaling_factor = volatility_scaling_factor
        
        self.enable_confidence_scaling = enable_confidence_scaling
        self.confidence_power = confidence_power
        
        self.enable_performance_scaling = enable_performance_scaling
        self.consecutive_loss_threshold = consecutive_loss_threshold
        self.consecutive_win_threshold = consecutive_win_threshold
        self.loss_scaling_factor = loss_scaling_factor
        self.win_scaling_factor = min(win_scaling_factor, 1.0)  # 上限 1.0
        self.max_scaling_factor = max_scaling_factor
        self.min_scaling_factor = min_scaling_factor
        
        # 历史表现追踪（在 metadata 中存储）
        self._consecutive_losses = 0
        self._consecutive_wins = 0
    
    def validate(self, ctx: RiskContext, order: OrderIntent) -> Tuple[bool, Optional[str]]:
        """
        验证订单是否可以通过
        
        Position Scaling 通常不拒绝订单，而是在 adjust 中调整仓位
        
        Args:
            ctx: 风险控制上下文
            order: 订单意图
            
        Returns:
            Tuple[bool, Optional[str]]: (是否通过, 拒绝原因)
        """
        # Position Scaling 不拒绝订单，只调整仓位
        return True, None
    
    def adjust(self, ctx: RiskContext, order: OrderIntent) -> OrderIntent:
        """
        调整订单仓位（基于多种缩放因子）
        
        Args:
            ctx: 风险控制上下文
            order: 订单意图
            
        Returns:
            OrderIntent: 调整后的订单意图
        """
        # 只调整买入订单（卖出订单不缩放）
        if order.side != OrderSide.BUY:
            return order
        
        # 计算综合缩放因子
        scaling_factor = self._calculate_scaling_factor(ctx, order)
        
        # 限制缩放因子范围
        scaling_factor = max(self.min_scaling_factor, min(scaling_factor, self.max_scaling_factor))
        
        # 如果缩放因子为 1.0，不需要调整
        if abs(scaling_factor - 1.0) < 1e-6:
            return order
        
        logger.info(
            f"[{self.name}] 调整仓位: {order.symbol}, "
            f"缩放因子={scaling_factor:.3f}"
        )
        
        # 创建调整后的订单
        adjusted_order = OrderIntent(
            symbol=order.symbol,
            side=order.side,
            timestamp=order.timestamp,
            qty=order.qty,
            target_weight=(
                order.target_weight * scaling_factor
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
                "position_scaling_adjusted": True,
                "original_weight": order.target_weight,
                "scaling_factor": scaling_factor,
                "scaling_components": self._get_scaling_components(ctx, order)
            }
        )
        
        # 如果使用 qty，也进行缩放
        if adjusted_order.qty is not None:
            adjusted_order.qty = max(1, int(adjusted_order.qty * scaling_factor))
        
        return adjusted_order
    
    def _calculate_scaling_factor(self, ctx: RiskContext, order: OrderIntent) -> float:
        """
        计算综合缩放因子
        
        Args:
            ctx: 风险控制上下文
            order: 订单意图
            
        Returns:
            float: 综合缩放因子
        """
        scaling_factors = []
        
        # 1. 账户权益缩放
        if self.enable_equity_scaling:
            equity_factor = self._calculate_equity_scaling(ctx)
            scaling_factors.append(equity_factor)
        
        # 2. 波动率缩放（需要额外的波动率数据）
        if self.enable_volatility_scaling:
            volatility_factor = self._calculate_volatility_scaling(ctx, order)
            scaling_factors.append(volatility_factor)
        
        # 3. 置信度缩放
        if self.enable_confidence_scaling:
            confidence_factor = self._calculate_confidence_scaling(order)
            scaling_factors.append(confidence_factor)
        
        # 4. 历史表现缩放
        if self.enable_performance_scaling:
            performance_factor = self._calculate_performance_scaling()
            scaling_factors.append(performance_factor)
        
        # 如果没有启用任何缩放，返回 1.0
        if not scaling_factors:
            return 1.0
        
        # 综合缩放因子：取最小值（最保守）
        # 或者可以改为乘积：return product(scaling_factors)
        # 这里使用最小值，确保最保守的风险控制
        return min(scaling_factors)
    
    def _calculate_equity_scaling(self, ctx: RiskContext) -> float:
        """
        计算账户权益缩放因子
        
        账户越大，单笔交易比例可能越小（风险分散）
        
        Args:
            ctx: 风险控制上下文
            
        Returns:
            float: 权益缩放因子
        """
        equity = ctx.get_account_equity()
        if equity <= 0:
            return 1.0
        
        # 如果账户权益小于基准，不缩放
        if equity <= self.base_equity:
            return 1.0
        
        # 账户越大，缩放因子越小（风险分散）
        # 公式：scaling = (base_equity / equity) ^ equity_scaling_factor
        ratio = self.base_equity / equity
        scaling = ratio ** self.equity_scaling_factor
        
        return scaling
    
    def _calculate_volatility_scaling(self, ctx: RiskContext, order: OrderIntent) -> float:
        """
        计算波动率缩放因子
        
        波动率越高，仓位越小（Kelly Criterion 变种）
        
        Args:
            ctx: 风险控制上下文
            order: 订单意图
            
        Returns:
            float: 波动率缩放因子
        """
        # 从 metadata 中获取波动率（如果可用）
        volatility = order.metadata.get("volatility")
        if volatility is None:
            # 如果没有波动率数据，返回 1.0（不缩放）
            return 1.0
        
        # 波动率越高，缩放因子越小
        # 公式：scaling = base_volatility / volatility * volatility_scaling_factor
        if volatility <= 0:
            return 1.0
        
        scaling = (self.base_volatility / volatility) ** self.volatility_scaling_factor
        return scaling
    
    def _calculate_confidence_scaling(self, order: OrderIntent) -> float:
        """
        计算置信度缩放因子
        
        置信度越低，仓位越小
        
        Args:
            order: 订单意图
            
        Returns:
            float: 置信度缩放因子
        """
        # 如果订单没有 confidence 字段，返回 1.0（不缩放）
        if order.confidence is None:
            return 1.0
        
        # 置信度缩放：confidence ^ confidence_power
        # confidence_power = 1.0: 线性
        # confidence_power = 2.0: 平方（更保守）
        scaling = order.confidence ** self.confidence_power
        
        return scaling
    
    def _calculate_performance_scaling(self) -> float:
        """
        计算历史表现缩放因子
        
        连续亏损后降低仓位，连续盈利后增加仓位
        
        Returns:
            float: 历史表现缩放因子
        """
        # 连续亏损：降低仓位
        if self._consecutive_losses >= self.consecutive_loss_threshold:
            return self.loss_scaling_factor
        
        # 连续盈利：增加仓位（上限 1.0）
        if self._consecutive_wins >= self.consecutive_win_threshold:
            return min(self.win_scaling_factor, 1.0)
        
        # 正常情况：不缩放
        return 1.0
    
    def _get_scaling_components(self, ctx: RiskContext, order: OrderIntent) -> Dict[str, float]:
        """
        获取各个缩放因子的详细值（用于调试）
        
        Args:
            ctx: 风险控制上下文
            order: 订单意图
            
        Returns:
            Dict[str, float]: 各个缩放因子的值
        """
        components = {}
        
        if self.enable_equity_scaling:
            components["equity"] = self._calculate_equity_scaling(ctx)
        
        if self.enable_volatility_scaling:
            components["volatility"] = self._calculate_volatility_scaling(ctx, order)
        
        if self.enable_confidence_scaling:
            components["confidence"] = self._calculate_confidence_scaling(order)
        
        if self.enable_performance_scaling:
            components["performance"] = self._calculate_performance_scaling()
        
        return components
    
    def post_trade(self, ctx: RiskContext, fills: List[Dict[str, Any]]):
        """
        交易后更新历史表现追踪
        
        Args:
            ctx: 风险控制上下文
            fills: 成交记录列表
        """
        if not self.enable_performance_scaling:
            return
        
        # 更新连续亏损/盈利计数
        # 这里简化处理，实际应该根据交易结果更新
        # 可以通过 metadata 传递交易结果
        for fill in fills:
            # 从 fill 中获取交易结果（如果可用）
            is_profitable = fill.get("is_profitable")
            if is_profitable is True:
                self._consecutive_wins += 1
                self._consecutive_losses = 0
            elif is_profitable is False:
                self._consecutive_losses += 1
                self._consecutive_wins = 0
            # 如果 is_profitable 为 None，不更新计数
    
    def __repr__(self):
        return (
            f"{self.__class__.__name__}(name='{self.name}', "
            f"equity_scaling={self.enable_equity_scaling}, "
            f"volatility_scaling={self.enable_volatility_scaling}, "
            f"confidence_scaling={self.enable_confidence_scaling}, "
            f"performance_scaling={self.enable_performance_scaling})"
        )

