"""
Layer B: Constrained Allocator（分配与风控层）
根据scores和风险预算生成目标权重，应用约束投影
"""
from typing import Dict, Optional
from trader.agent.TradingAgent import TradingAgent
from trader.backtest.engine import BacktestEngine
from trader.logger import get_logger
from trader.agent.theta import Theta
import numpy as np

logger = get_logger(__name__)


class ConstrainedAllocator(TradingAgent):
    """
    约束分配器：根据scores和风险预算生成目标权重
    
    核心逻辑：
    1. 初始权重：w~_i(t) ∝ max(score_i(t), 0) / σ_i(t)
    2. 约束投影：
       - ∑|w_i| ≤ gross_exposure
       - 0 ≤ w_i ≤ max_w (long-only)
       - ∑|w_i(t) - w_i(t-1)| ≤ turnover_cap
    3. risk_mode缩放总仓位
    """
    
    def __init__(
        self,
        signal_layer: TradingAgent,
        theta: Theta,
        name: str = "ConstrainedAllocator"
    ):
        """
        初始化约束分配器
        
        Args:
            signal_layer: 信号层Agent（用于获取scores）
            theta: 参数θ
            name: Agent名称
        """
        super().__init__(
            name=name,
            max_position_weight=theta.max_w,
            min_score_threshold=theta.enter_th,
            max_total_weight=theta.gross_exposure
        )
        
        self.signal_layer = signal_layer
        self.theta = theta
        
        # 记录上一期的权重（用于turnover约束）
        self.prev_weights: Dict[str, float] = {}
    
    def score(self, stock_code: str, engine: BacktestEngine) -> float:
        """
        计算股票的score（代理到signal_layer）
        
        Args:
            stock_code: 股票代码
            engine: 回测引擎
            
        Returns:
            float: score
        """
        return self.signal_layer.score(stock_code, engine)
    
    def _get_risk_adjusted_scores(
        self,
        scores: Dict[str, float],
        engine: BacktestEngine
    ) -> Dict[str, float]:
        """
        计算风险调整后的scores
        
        w~_i(t) ∝ max(score_i(t), 0) / σ_i(t)
        
        Args:
            scores: {stock_code: score}
            engine: 回测引擎
            
        Returns:
            Dict: {stock_code: risk_adjusted_score}
        """
        risk_adjusted = {}
        
        for stock_code, score in scores.items():
            if score <= 0:
                risk_adjusted[stock_code] = 0.0
                continue
            
            try:
                # 获取波动率
                vol_20d = engine.get_feature("vol_20d", stock_code)
                if vol_20d is None or vol_20d <= 0:
                    # 如果没有波动率数据，使用默认值
                    vol_20d = 0.02  # 假设2%的日波动率
                
                # 风险调整：score / volatility
                risk_adjusted[stock_code] = score / vol_20d
            except Exception as e:
                logger.debug(f"计算风险调整score失败 {stock_code}: {e}")
                risk_adjusted[stock_code] = score
        
        return risk_adjusted
    
    def _apply_constraints(
        self,
        raw_weights: Dict[str, float]
    ) -> Dict[str, float]:
        """
        应用约束投影
        
        约束：
        1. ∑|w_i| ≤ gross_exposure
        2. 0 ≤ w_i ≤ max_w
        3. ∑|w_i(t) - w_i(t-1)| ≤ turnover_cap
        
        Args:
            raw_weights: 原始权重 {stock_code: weight}
            
        Returns:
            Dict: 约束后的权重 {stock_code: weight}
        """
        if not raw_weights:
            return {}
        
        # 转换为numpy数组便于计算
        stock_codes = list(raw_weights.keys())
        weights = np.array([raw_weights[code] for code in stock_codes])
        
        # 约束1: 0 ≤ w_i ≤ max_w
        weights = np.clip(weights, 0.0, self.theta.max_w)
        
        # 约束2: ∑|w_i| ≤ gross_exposure
        total_weight = np.sum(np.abs(weights))
        if total_weight > self.theta.gross_exposure:
            # 按比例缩放
            scale = self.theta.gross_exposure / total_weight
            weights = weights * scale
        
        # 约束3: ∑|w_i(t) - w_i(t-1)| ≤ turnover_cap
        if self.prev_weights:
            prev_weights_array = np.array([
                self.prev_weights.get(code, 0.0) for code in stock_codes
            ])
            
            # 计算换手
            turnover = np.sum(np.abs(weights - prev_weights_array))
            
            if turnover > self.theta.turnover_cap:
                # 需要限制换手
                # 使用简单的缩放方法：按比例缩小变化量
                max_change = self.theta.turnover_cap
                current_change = turnover
                
                if current_change > 0:
                    # 计算可以允许的变化比例
                    change_scale = max_change / current_change
                    
                    # 调整权重：w_new = w_prev + (w_target - w_prev) * scale
                    weights = prev_weights_array + (weights - prev_weights_array) * change_scale
                    
                    # 重新应用约束1和2
                    weights = np.clip(weights, 0.0, self.theta.max_w)
                    total_weight = np.sum(np.abs(weights))
                    if total_weight > self.theta.gross_exposure:
                        scale = self.theta.gross_exposure / total_weight
                        weights = weights * scale
        
        # 应用risk_mode缩放
        risk_scale = self.theta.get_risk_scale()
        weights = weights * risk_scale
        
        # 转换回字典
        constrained_weights = {
            code: float(w) for code, w in zip(stock_codes, weights)
        }
        
        # 更新prev_weights
        self.prev_weights = constrained_weights.copy()
        
        return constrained_weights
    
    def get_weights(
        self,
        stock_codes: list,
        engine: BacktestEngine
    ) -> Dict[str, float]:
        """
        获取所有股票的目标权重（应用约束后）
        
        Args:
            stock_codes: 股票代码列表
            engine: 回测引擎
            
        Returns:
            Dict: {stock_code: weight} 已归一化和约束的权重
        """
        # 1. 从信号层获取scores
        scores = {}
        for stock_code in stock_codes:
            try:
                score = self.signal_layer.score(stock_code, engine)
                scores[stock_code] = score
            except Exception as e:
                logger.warning(f"获取score失败 {stock_code}: {e}")
                scores[stock_code] = 0.0
        
        # 2. 风险调整
        risk_adjusted_scores = self._get_risk_adjusted_scores(scores, engine)
        
        # 3. 归一化（使总和为1）
        total_score = sum(risk_adjusted_scores.values())
        if total_score <= 0:
            return {code: 0.0 for code in stock_codes}
        
        raw_weights = {
            code: score / total_score
            for code, score in risk_adjusted_scores.items()
        }
        
        # 4. 应用约束
        constrained_weights = self._apply_constraints(raw_weights)
        
        return constrained_weights
    
    def update_theta(self, theta: Theta):
        """更新参数θ"""
        self.theta = theta
        self.max_position_weight = theta.max_w
        self.max_total_weight = theta.gross_exposure
        self.min_score_threshold = theta.enter_th
        logger.info(f"[{self.name}] 更新参数θ: {theta}")
    
    def reset_prev_weights(self):
        """重置上一期权重（用于新周期开始）"""
        self.prev_weights = {}

