"""
多资产交易代理（使用 TurtleAgent）
对多个股票分别使用独立的 TurtleAgent 获取信号，然后进行权重归一化
支持并行计算以提升性能
"""
from typing import Dict, List, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
from trader.agent.TradingAgent import TradingAgent
from trader.agent.agent_turtle import TurtleAgent
from trader.agent.multiagent_weight_normalized import normalize_weights
from trader.backtest.engine import BacktestEngine
from trader.logger import get_logger
import os

logger = get_logger(__name__)


class MultiAssetTurtleAgent(TradingAgent):
    """
    多资产交易代理（使用 TurtleAgent）
    对每支股票使用独立的 TurtleAgent 获取信号，然后进行权重归一化
    """
    
    def __init__(
        self,
        stock_codes: List[str],
        name: str = "MultiAssetTurtleAgent",
        max_position_weight: float = 0.2,
        min_score_threshold: float = 0.0,
        max_total_weight: float = 1.0,
        # TurtleAgent 参数
        entry_period: int = 20,
        exit_period: int = 10,
        atr_period: int = 20,
        risk_per_trade: float = 0.02,
        stop_loss_atr: float = 2.0,
        max_positions: int = 4,
        add_position_atr: float = 0.5,
        # 并行计算参数
        use_parallel: bool = False,
        max_workers: Optional[int] = None
    ):
        """
        初始化多资产交易代理（使用 TurtleAgent）
        
        Args:
            stock_codes: 股票代码列表
            name: Agent 名称
            max_position_weight: 单个股票最大配置比例
            min_score_threshold: 最小 score 阈值
            max_total_weight: 总配置比例上限
            entry_period: TurtleAgent 突破周期
            exit_period: TurtleAgent 退出周期
            atr_period: TurtleAgent ATR计算周期
            risk_per_trade: TurtleAgent 每次交易风险
            stop_loss_atr: TurtleAgent 止损距离（ATR倍数）
            max_positions: TurtleAgent 最大加仓次数
            add_position_atr: TurtleAgent 加仓距离（ATR倍数）
            use_parallel: 是否使用并行计算
            max_workers: 最大工作线程数，None 表示自动
        """
        super().__init__(
            name=name,
            max_position_weight=max_position_weight,
            min_score_threshold=min_score_threshold,
            max_total_weight=max_total_weight
        )
        
        self.stock_codes = stock_codes
        self.use_parallel = use_parallel
        
        # 设置并行计算的线程数
        if max_workers is None:
            cpu_count = os.cpu_count() or 4
            self.max_workers = min(len(stock_codes), cpu_count)
        else:
            self.max_workers = max_workers
        
        # 为每支股票创建独立的 TurtleAgent
        self.agents: Dict[str, TurtleAgent] = {}
        for stock_code in stock_codes:
            self.agents[stock_code] = TurtleAgent(
                name=f"{name}_{stock_code}",
                entry_period=entry_period,
                exit_period=exit_period,
                atr_period=atr_period,
                risk_per_trade=risk_per_trade,
                stop_loss_atr=stop_loss_atr,
                max_positions=max_positions,
                add_position_atr=add_position_atr
            )
            # 设置交易股票列表
            self.agents[stock_code].set_trading_stocks([stock_code])
        
        parallel_info = f", 并行计算: {self.max_workers} 线程" if use_parallel else ", 顺序计算"
        logger.info(
            f"创建 MultiAssetTurtleAgent: {len(stock_codes)} 支股票, "
            f"每支股票使用独立的 TurtleAgent{parallel_info}"
        )
    
    def score(self, stock_code: str, engine: BacktestEngine) -> float:
        """
        计算股票的 score（使用该股票对应的 TurtleAgent）
        
        Args:
            stock_code: 股票代码
            engine: 回测引擎
            
        Returns:
            float: 分数
        """
        if stock_code not in self.agents:
            logger.warning(f"股票 {stock_code} 不在代理列表中")
            return 0.0
        
        agent = self.agents[stock_code]
        return agent.score(stock_code, engine)
    
    def get_all_scores(self, engine: BacktestEngine) -> Dict[str, float]:
        """
        获取所有股票的 score（支持并行计算）
        
        Args:
            engine: 回测引擎
            
        Returns:
            Dict[str, float]: {stock_code: score} 分数字典
        """
        if not self.use_parallel or len(self.stock_codes) <= 1:
            scores = {}
            for stock_code in self.stock_codes:
                scores[stock_code] = self.score(stock_code, engine)
            return scores
        
        # 并行计算
        scores = {}
        
        def _compute_score(stock_code: str) -> tuple[str, float]:
            """计算单个股票的 score"""
            try:
                score = self.score(stock_code, engine)
                return (stock_code, score)
            except Exception as e:
                logger.error(f"计算 {stock_code} 的 score 时出错: {e}", exc_info=True)
                return (stock_code, 0.0)
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_stock = {
                executor.submit(_compute_score, stock_code): stock_code
                for stock_code in self.stock_codes
            }
            
            for future in as_completed(future_to_stock):
                stock_code, score = future.result()
                scores[stock_code] = score
        
        return scores
    
    def get_all_weights(self, engine: BacktestEngine) -> Dict[str, float]:
        """
        获取所有股票的权重（归一化后，支持并行计算）
        
        Args:
            engine: 回测引擎
            
        Returns:
            Dict[str, float]: {stock_code: weight} 权重字典（已归一化）
        """
        # 获取所有股票的 score（已并行化）
        scores = self.get_all_scores(engine)
        
        if not self.use_parallel or len(scores) <= 1:
            # 顺序计算权重
            raw_weights = {}
            for stock_code, score in scores.items():
                agent = self.agents[stock_code]
                raw_weights[stock_code] = agent.weight(stock_code, score, engine)
        else:
            # 并行计算权重
            raw_weights = {}
            
            def _compute_weight(stock_code: str, score: float) -> tuple[str, float]:
                """计算单个股票的权重"""
                try:
                    agent = self.agents[stock_code]
                    weight = agent.weight(stock_code, score, engine)
                    return (stock_code, weight)
                except Exception as e:
                    logger.error(f"计算 {stock_code} 的 weight 时出错: {e}", exc_info=True)
                    return (stock_code, 0.0)
            
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                future_to_stock = {
                    executor.submit(_compute_weight, stock_code, score): stock_code
                    for stock_code, score in scores.items()
                }
                
                for future in as_completed(future_to_stock):
                    stock_code, weight = future.result()
                    raw_weights[stock_code] = weight
        
        # 归一化权重
        normalized_weights = normalize_weights(raw_weights, self.max_total_weight)
        
        return normalized_weights
    
    def on_date(self, engine: BacktestEngine, date: str):
        """
        每个交易日的回调函数（更新所有 agent 的状态，支持并行计算）
        
        Args:
            engine: 回测引擎
            date: 当前日期
        """
        if not self.use_parallel or len(self.agents) <= 1:
            # 顺序更新
            for agent in self.agents.values():
                agent.on_date(engine, date)
        else:
            # 并行更新
            def _update_agent(agent):
                """更新单个 agent 的状态"""
                try:
                    agent.on_date(engine, date)
                except Exception as e:
                    logger.error(f"更新 agent {agent.name} 状态时出错: {e}", exc_info=True)
            
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                futures = [
                    executor.submit(_update_agent, agent)
                    for agent in self.agents.values()
                ]
                
                for future in as_completed(futures):
                    future.result()  # 检查是否有异常

