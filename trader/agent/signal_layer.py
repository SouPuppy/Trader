"""
Layer A: Signal Layer（信号层）
组合trends/fundamentals/news生成统一score
"""
from typing import Dict, Optional
from trader.agent.TradingAgent import TradingAgent
from trader.backtest.engine import BacktestEngine
from trader.logger import get_logger
from trader.agent.theta import Theta

logger = get_logger(__name__)


class SignalLayer(TradingAgent):
    """
    信号层：组合多个因子生成统一score
    
    score_i(t) = w_T·Trend_i(t) + w_V·VolAdj_i(t) + w_F·Value_i(t) + w_N·News_i(t)
    """
    
    def __init__(
        self,
        name: str = "SignalLayer",
        # 因子权重（基础权重）
        w_T: float = 0.4,  # Trend权重
        w_V: float = 0.2,  # Volatility权重
        w_F: float = 0.2,  # Fundamental权重
        w_N: float = 0.2,  # News权重
        # Theta参数（用于微调）
        theta: Optional[Theta] = None
    ):
        """
        初始化信号层
        
        Args:
            name: Agent名称
            w_T: Trend权重
            w_V: Volatility权重
            w_F: Fundamental权重
            w_N: News权重
            theta: 参数θ（用于微调权重）
        """
        super().__init__(
            name=name,
            max_position_weight=1.0,  # 信号层不限制仓位，由分配层控制
            min_score_threshold=0.0,
            max_total_weight=1.0
        )
        
        self.base_w_T = w_T
        self.base_w_V = w_V
        self.base_w_F = w_F
        self.base_w_N = w_N
        
        # 归一化权重
        total = w_T + w_V + w_F + w_N
        if total > 0:
            self.base_w_T /= total
            self.base_w_V /= total
            self.base_w_F /= total
            self.base_w_N /= total
        
        self.theta = theta or Theta()
    
    def _get_effective_weights(self) -> Dict[str, float]:
        """
        获取有效权重（基础权重 + theta微调）
        
        Returns:
            Dict: {w_T, w_V, w_F, w_N}
        """
        w_T = self.base_w_T
        w_V = self.base_w_V
        w_F = self.base_w_F
        w_N = self.base_w_N
        
        # 应用theta的微调
        if self.theta.factor_weights_delta:
            w_T += self.theta.factor_weights_delta.get("w_T", 0.0)
            w_V += self.theta.factor_weights_delta.get("w_V", 0.0)
            w_F += self.theta.factor_weights_delta.get("w_F", 0.0)
            w_N += self.theta.factor_weights_delta.get("w_N", 0.0)
        
        # 归一化
        total = w_T + w_V + w_F + w_N
        if total > 0:
            w_T /= total
            w_V /= total
            w_F /= total
            w_N /= total
        
        return {"w_T": w_T, "w_V": w_V, "w_F": w_F, "w_N": w_N}
    
    def _get_trend_signal(self, stock_code: str, engine: BacktestEngine) -> float:
        """
        获取趋势信号（使用多时间框架）
        
        Returns:
            float: 趋势信号 [-1, 1]
        """
        try:
            # 获取多个时间框架的收益率
            ret_1d = engine.get_feature("ret_1d", stock_code) or 0.0
            ret_5d = engine.get_feature("ret_5d", stock_code) or 0.0
            ret_20d = engine.get_feature("ret_20d", stock_code) or 0.0
            
            # 加权组合：短期权重低，中期和长期权重高
            # 这样可以捕捉趋势的持续性
            trend_score = (
                0.2 * ret_1d / 0.05 +      # 1日收益率，归一化到[-1, 1]，假设最大5%
                0.3 * ret_5d / 0.10 +      # 5日收益率，归一化到[-1, 1]，假设最大10%
                0.5 * ret_20d / 0.30       # 20日收益率，归一化到[-1, 1]，假设最大30%
            )
            
            # 裁剪到[-1, 1]
            trend_score = max(-1.0, min(1.0, trend_score))
            return trend_score
        except Exception as e:
            logger.debug(f"获取趋势信号失败 {stock_code}: {e}")
            return 0.0
    
    def _get_volatility_signal(self, stock_code: str, engine: BacktestEngine) -> float:
        """
        获取波动率调整信号（低波动率 = 正信号）
        
        使用相对波动率：与市场平均波动率比较，而不是绝对波动率
        
        Returns:
            float: 波动率信号 [-1, 1]，低波动率为正
        """
        try:
            vol_20d = engine.get_feature("vol_20d", stock_code)
            if vol_20d is None or vol_20d <= 0:
                return 0.0
            
            # 获取60日波动率作为长期基准
            vol_60d = engine.get_feature("vol_60d", stock_code)
            
            if vol_60d and vol_60d > 0:
                # 使用相对波动率：20日波动率相对于60日波动率
                # 如果20日波动率低于60日波动率，说明波动率在下降，是正信号
                vol_ratio = vol_20d / vol_60d
                
                # vol_ratio < 1 表示波动率下降（正信号）
                # vol_ratio > 1 表示波动率上升（负信号）
                if vol_ratio < 0.7:
                    signal = 1.0  # 波动率显著下降
                elif vol_ratio < 0.9:
                    signal = 0.5  # 波动率下降
                elif vol_ratio < 1.1:
                    signal = 0.0  # 波动率稳定
                elif vol_ratio < 1.5:
                    signal = -0.5  # 波动率上升
                else:
                    signal = -1.0  # 波动率显著上升
                
                return signal
            else:
                # 如果没有60日波动率，使用绝对波动率
                # 归一化波动率，假设在[0, 0.1]范围内
                vol_normalized = max(0.0, min(1.0, vol_20d / 0.1))
                # 转换为信号：低波动率 = 正信号
                signal = 1.0 - vol_normalized
                return signal * 2.0 - 1.0  # 转换到[-1, 1]
        except Exception as e:
            logger.debug(f"获取波动率信号失败 {stock_code}: {e}")
            return 0.0
    
    def _get_fundamental_signal(self, stock_code: str, engine: BacktestEngine) -> float:
        """
        获取基本面信号（基于PE/PB/PS等综合指标）
        
        Returns:
            float: 基本面信号 [-1, 1]
        """
        try:
            # 获取多个估值指标
            pe_ratio = engine.get_feature("pe_ratio_ttm", stock_code)
            if pe_ratio is None:
                pe_ratio = engine.get_feature("pe_ratio", stock_code)
            
            pb_ratio = engine.get_feature("pb_ratio", stock_code)
            ps_ratio = engine.get_feature("ps_ratio_ttm", stock_code)
            if ps_ratio is None:
                ps_ratio = engine.get_feature("ps_ratio", stock_code)
            
            signals = []
            
            # PE信号：PE越低越好（假设合理PE在10-30之间）
            if pe_ratio is not None and pe_ratio > 0:
                if pe_ratio < 10:
                    pe_signal = 1.0  # 非常便宜
                elif pe_ratio < 15:
                    pe_signal = 0.7  # 便宜
                elif pe_ratio < 20:
                    pe_signal = 0.3  # 合理偏低
                elif pe_ratio < 30:
                    pe_signal = -0.3  # 偏高
                else:
                    pe_signal = -1.0  # 很贵
                signals.append(pe_signal)
            
            # PB信号：PB越低越好（假设合理PB在1-5之间）
            if pb_ratio is not None and pb_ratio > 0:
                if pb_ratio < 1:
                    pb_signal = 1.0  # 非常便宜
                elif pb_ratio < 2:
                    pb_signal = 0.5  # 合理
                elif pb_ratio < 5:
                    pb_signal = -0.5  # 偏高
                else:
                    pb_signal = -1.0  # 很贵
                signals.append(pb_signal)
            
            # PS信号：PS越低越好（假设合理PS在1-10之间）
            if ps_ratio is not None and ps_ratio > 0:
                if ps_ratio < 1:
                    ps_signal = 1.0  # 非常便宜
                elif ps_ratio < 3:
                    ps_signal = 0.5  # 合理
                elif ps_ratio < 10:
                    ps_signal = -0.5  # 偏高
                else:
                    ps_signal = -1.0  # 很贵
                signals.append(ps_signal)
            
            if not signals:
                return 0.0
            
            # 取平均值
            return sum(signals) / len(signals)
        except Exception as e:
            logger.debug(f"获取基本面信号失败 {stock_code}: {e}")
            return 0.0
    
    def _get_news_signal(self, stock_code: str, engine: BacktestEngine) -> float:
        """
        获取新闻信号（使用最基础的features，朴素算法）
        
        只使用数据库中最基础的新闻统计特征（新闻数量），不做任何复杂分析
        使用 engine.get_feature() 获取已计算好的特征值
        
        Returns:
            float: 新闻信号 [0, 1]，基于新闻数量（新闻越多，信号越强）
        """
        try:
            # 使用最基础的新闻特征：新闻数量
            # 优先使用5日新闻数量（如果存在），否则使用当日新闻数量
            news_count = engine.get_feature("news_count_5d", stock_code)
            
            if news_count is None:
                # 如果没有5日特征，尝试当日新闻数量
                news_count = engine.get_feature("news_count", stock_code)
                if news_count is None or news_count == 0:
                    return 0.0
                # 使用当日新闻数量，简单归一化
                # 假设0-10条新闻为正常范围，超过10条为高关注度
                news_signal = min(1.0, news_count / 10.0)  # 归一化到[0, 1]
                return news_signal
            else:
                # 使用5日新闻数量，简单归一化
                # 假设0-30条新闻为正常范围（5天）
                if news_count == 0:
                    return 0.0
                news_signal = min(1.0, news_count / 30.0)  # 归一化到[0, 1]
                return news_signal
            
        except Exception as e:
            logger.debug(f"获取新闻信号失败 {stock_code}: {e}")
            return 0.0
    
    def score(self, stock_code: str, engine: BacktestEngine) -> float:
        """
        计算股票的score
        
        score_i(t) = w_T·Trend_i(t) + w_V·VolAdj_i(t) + w_F·Value_i(t) + w_N·News_i(t)
        
        Args:
            stock_code: 股票代码
            engine: 回测引擎
            
        Returns:
            float: score [-1, 1]
        """
        weights = self._get_effective_weights()
        
        trend = self._get_trend_signal(stock_code, engine)
        vol = self._get_volatility_signal(stock_code, engine)
        fundamental = self._get_fundamental_signal(stock_code, engine)
        news = self._get_news_signal(stock_code, engine)
        
        score = (
            weights["w_T"] * trend +
            weights["w_V"] * vol +
            weights["w_F"] * fundamental +
            weights["w_N"] * news
        )
        
        # 应用进出场阈值
        if score < self.theta.exit_th:
            score = 0.0  # 低于出场阈值，不持有
        elif score < self.theta.enter_th:
            score = 0.0  # 低于进场阈值，不买入
        
        return score
    
    def update_theta(self, theta: Theta):
        """更新参数θ"""
        self.theta = theta
        logger.info(f"[{self.name}] 更新参数θ: {theta}")

