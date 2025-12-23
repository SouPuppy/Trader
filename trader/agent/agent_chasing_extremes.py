"""
Chasing Extremes Agent - 追逐极端价格策略
这是一个"疯狂的" agent，用于测试 risk control 是否有用

策略逻辑：
- 当价格出现极端波动（大涨或大跌）时，会追逐这个趋势
- 追涨：当价格大幅上涨时，全仓买入
- 追跌：当价格大幅下跌时，全仓卖出
- 没有风险控制，会全仓或大仓位买入/卖出
"""
from typing import Dict, Optional
from trader.agent.TradingAgent import TradingAgent
from trader.backtest.engine import BacktestEngine
from trader.logger import get_logger

logger = get_logger(__name__)


class ChasingExtremesAgent(TradingAgent):
    """
    追逐极端价格策略 Agent
    
    这是一个"疯狂的" agent，用于测试 risk control 是否有用
    当价格出现极端波动时，会全仓追逐趋势
    """
    
    def __init__(
        self,
        name: str = "ChasingExtremesAgent",
        extreme_threshold: float = 0.05,  # 极端波动阈值（5%）
        lookback_days: int = 1,  # 回看天数，用于计算涨跌幅
        max_position_weight: float = 1.0,  # 最大仓位（全仓）
        min_score_threshold: float = 0.0,
        max_total_weight: float = 1.0,
        chase_up: bool = True,  # 是否追涨
        chase_down: bool = True,  # 是否追跌
    ):
        """
        初始化 Chasing Extremes Agent
        
        Args:
            name: Agent 名称
            extreme_threshold: 极端波动阈值（如 0.05 表示 5%）
            lookback_days: 回看天数，用于计算涨跌幅
            max_position_weight: 最大仓位（默认全仓）
            min_score_threshold: 最小 score 阈值
            max_total_weight: 总配置比例上限（默认全仓）
            chase_up: 是否追涨（价格大涨时买入）
            chase_down: 是否追跌（价格大跌时卖出）
        """
        super().__init__(
            name=name,
            max_position_weight=max_position_weight,
            min_score_threshold=min_score_threshold,
            max_total_weight=max_total_weight
        )
        self.extreme_threshold = extreme_threshold
        self.lookback_days = lookback_days
        self.chase_up = chase_up
        self.chase_down = chase_down
        
        # 记录历史价格（用于计算涨跌幅）
        self.price_history: Dict[str, list] = {}  # {stock_code: [(date, price), ...]}
    
    def _update_price_history(self, stock_code: str, date: str, price: float):
        """更新价格历史"""
        if stock_code not in self.price_history:
            self.price_history[stock_code] = []
        
        self.price_history[stock_code].append((date, price))
        
        # 只保留必要的历史数据
        max_history = self.lookback_days + 5
        if len(self.price_history[stock_code]) > max_history:
            self.price_history[stock_code] = self.price_history[stock_code][-max_history:]
    
    def _calculate_return(self, stock_code: str, date: str) -> Optional[float]:
        """
        计算最近 lookback_days 天的收益率
        
        Args:
            stock_code: 股票代码
            date: 当前日期
            
        Returns:
            收益率，如果数据不足则返回 None
        """
        history = self.price_history.get(stock_code, [])
        if len(history) < self.lookback_days + 1:
            return None
        
        # 获取当前价格和 lookback_days 天前的价格
        current_price = history[-1][1]
        past_price = history[-(self.lookback_days + 1)][1]
        
        if past_price == 0 or current_price is None or past_price is None:
            return None
        
        return (current_price / past_price) - 1.0
    
    def score(self, stock_code: str, engine: BacktestEngine) -> float:
        """
        计算股票的 score（基于极端价格波动）
        
        策略（稳定亏钱的疯狂策略 - 反向操作）：
        - 如果价格上涨（超过阈值），返回负分（卖出信号，在上涨时卖出）
        - 如果价格下跌（超过阈值），返回正分（买入信号，在下跌时买入）
        - 反向操作，让它总是在错误的时间交易，稳定亏钱
        - 即使波动很小也会触发，让它频繁交易亏钱
        
        Args:
            stock_code: 股票代码
            engine: 回测引擎
            
        Returns:
            float: score 值，范围 [-1, 1]
                  - 正数表示买入信号（在下跌时买入）
                  - 负数表示卖出信号（在上涨时卖出）
                  - 绝对值越大表示波动越极端
        """
        try:
            date = engine.current_date
            if date is None:
                return 0.0
            
            # 获取当前价格
            current_price = engine.get_current_price(stock_code)
            if current_price is None or current_price == 0:
                return 0.0
            
            # 更新价格历史
            self._update_price_history(stock_code, date, current_price)
            
            # 计算收益率
            ret = self._calculate_return(stock_code, date)
            if ret is None:
                # 如果数据不足（第一天），返回一个正分，让它买入
                return 0.5 if self.chase_up else 0.0
            
            # 判断是否为极端波动
            abs_ret = abs(ret)
            
            # 反向操作：价格上涨时卖出，价格下跌时买入（稳定亏钱）
            # 降低阈值要求，让它更容易触发
            if abs_ret < self.extreme_threshold:
                # 即使波动不够极端，如果方向明确，也给予小的信号
                if ret > 0 and self.chase_up and abs_ret > self.extreme_threshold * 0.5:
                    # 价格上涨时卖出（反向操作，稳定亏钱）
                    return -min(abs_ret / self.extreme_threshold, 0.5)
                elif ret < 0 and self.chase_down and abs_ret > self.extreme_threshold * 0.5:
                    # 价格下跌时买入（反向操作，稳定亏钱）
                    return min(abs_ret / self.extreme_threshold, 0.5)
                else:
                    return 0.0
            
            # 极端波动：反向操作
            if ret > 0 and self.chase_up:
                # 大涨：反向操作，返回负分（在上涨时卖出，稳定亏钱）
                normalized_score = -min(abs_ret / (self.extreme_threshold * 2), 1.0)
                return normalized_score
            elif ret < 0 and self.chase_down:
                # 大跌：反向操作，返回正分（在下跌时买入，稳定亏钱）
                normalized_score = min(abs_ret / (self.extreme_threshold * 2), 1.0)
                return normalized_score
            
            return 0.0
            
        except Exception as e:
            logger.error(f"计算 {stock_code} 的 score 时出错: {e}", exc_info=True)
            return 0.0
    
    def weight(self, stock_code: str, score: float, engine: BacktestEngine) -> float:
        """
        计算实际资金配置比例（疯狂的策略：全仓或大仓位）
        
        这个 agent 没有风险控制，会全仓买入/卖出
        
        Args:
            stock_code: 股票代码
            score: 该股票的 score 值
            engine: 回测引擎
            
        Returns:
            float: 资金配置比例 [0, max_position_weight]
        """
        # 如果 score 为 0，不配置
        if abs(score) < self.min_score_threshold:
            return 0.0
        
        # 疯狂的策略：总是全仓或大仓位，让它稳定亏钱
        # 即使 score 很小，也给予较大的仓位
        abs_score = abs(score)
        
        # 将 score 映射到仓位，但给予最小仓位保证
        # score = 1.0 -> weight = max_position_weight (全仓)
        # score = 0.1 -> weight = max_position_weight * 0.5 (至少半仓，让它频繁交易)
        # 即使 score 很小，也给予至少 50% 的仓位
        min_weight = self.max_position_weight * 0.5  # 最小50%仓位
        weight = max(abs_score * self.max_position_weight, min_weight)
        
        # 限制在 [0, max_position_weight] 范围内
        weight = min(weight, self.max_position_weight)
        weight = max(0.0, weight)
        
        return weight
    
    def on_date(self, engine: BacktestEngine, date: str):
        """
        每个交易日的回调函数
        
        这个 agent 会疯狂地追逐极端价格，没有风险控制
        
        Args:
            engine: 回测引擎
            date: 当前日期
        """
        # 这个 agent 主要通过 score/weight 接口工作
        # 如果需要额外的逻辑，可以在这里实现
        pass

