"""
多资产交易代理（使用 LogisticAgentWithRAGGate）
对多个股票分别使用独立的 LogisticAgentWithRAGGate 获取信号
然后使用 RAG Gate 进行风险控制，最后进行权重归一化
支持并行计算以提升性能
"""
from typing import Dict, List, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
from trader.agent.TradingAgent import TradingAgent
from trader.agent.agent_logistic_with_rag_gate import LogisticAgentWithRAGGate
from trader.agent.multiagent_weight_normalized import normalize_weights
from trader.backtest.engine import BacktestEngine
from trader.logger import get_logger
import os

logger = get_logger(__name__)


class MultiAssetLogisticAgentWithRAGGate(TradingAgent):
    """
    多资产交易代理（使用 LogisticAgentWithRAGGate）
    对每支股票使用独立的 LogisticAgentWithRAGGate 获取信号
    然后使用 RAG Gate 进行风险控制，最后进行权重归一化
    """
    
    def __init__(
        self,
        stock_codes: List[str],
        name: str = "MultiAssetLogisticAgentWithRAGGate",
        max_position_weight: float = 0.2,
        min_score_threshold: float = 0.0,
        max_total_weight: float = 1.0,
        # LogisticAgent 参数
        feature_names: Optional[List[str]] = None,
        train_window_days: int = 252,
        prediction_horizon: int = 5,
        ret_threshold: float = 0.0,
        retrain_frequency: int = 20,
        train_test_split_ratio: float = 0.7,
        # RAG Gate 参数
        llm_model: str = "deepseek-chat",
        llm_temperature: float = 0.3,
        test_mode: bool = False,
        test_force_reject: bool = False,
        # 并行计算参数
        use_parallel: bool = False,
        max_workers: Optional[int] = None
    ):
        """
        初始化多资产交易代理
        
        Args:
            stock_codes: 股票代码列表
            name: Agent 名称
            max_position_weight: 单个股票最大配置比例
            min_score_threshold: 最小 score 阈值
            max_total_weight: 总配置比例上限
            feature_names: LogisticAgent 使用的特征名称列表
            train_window_days: LogisticAgent 训练窗口大小
            prediction_horizon: LogisticAgent 预测周期
            ret_threshold: LogisticAgent 收益阈值
            retrain_frequency: LogisticAgent 重新训练频率
            train_test_split_ratio: LogisticAgent 训练/测试分割比例
            llm_model: LLM 模型名称（RAG 系统内部使用）
            llm_temperature: LLM 温度参数（控制随机性）
            test_mode: 测试模式，如果为 True，会打印更详细的调试信息
            test_force_reject: 测试模式下的强制拒绝，如果为 True，所有 gate 评估都会返回 False
            use_parallel: 是否使用并行计算
            max_workers: 最大工作线程数，None 表示自动（默认为股票数量）
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
        
        # 为每支股票创建独立的 LogisticAgentWithRAGGate
        self.agents: Dict[str, LogisticAgentWithRAGGate] = {}
        for stock_code in stock_codes:
            self.agents[stock_code] = LogisticAgentWithRAGGate(
                name=f"{name}_{stock_code}",
                feature_names=feature_names,
                train_window_days=train_window_days,
                prediction_horizon=prediction_horizon,
                ret_threshold=ret_threshold,
                retrain_frequency=retrain_frequency,
                max_position_weight=max_position_weight,
                min_score_threshold=min_score_threshold,
                max_total_weight=max_total_weight,
                train_test_split_ratio=train_test_split_ratio,
                llm_model=llm_model,
                llm_temperature=llm_temperature,
                test_mode=test_mode,
                test_force_reject=test_force_reject
            )
        
        parallel_info = f", 并行计算: {self.max_workers} 线程" if use_parallel else ", 顺序计算"
        logger.info(
            f"创建 MultiAssetLogisticAgentWithRAGGate: {len(stock_codes)} 支股票, "
            f"每支股票使用独立的 LogisticAgentWithRAGGate{parallel_info}"
        )
    
    def score(self, stock_code: str, engine: BacktestEngine) -> float:
        """
        计算股票的 score（使用该股票对应的 LogisticAgentWithRAGGate）
        
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
            # 顺序计算（单股票或禁用并行）
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
            # 提交所有任务
            future_to_stock = {
                executor.submit(_compute_score, stock_code): stock_code
                for stock_code in self.stock_codes
            }
            
            # 收集结果
            for future in as_completed(future_to_stock):
                stock_code, score = future.result()
                scores[stock_code] = score
        
        return scores
    
    def get_all_weights(
        self, 
        engine: BacktestEngine,
        apply_rag_gate: bool = True
    ) -> Dict[str, float]:
        """
        获取所有股票的权重（归一化后，支持并行计算）
        在计算权重后，使用 RAG Gate 进行风险控制
        
        Args:
            engine: 回测引擎
            apply_rag_gate: 是否应用 RAG Gate 风险控制
            
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
                weight = agent.weight(stock_code, score, engine)
                
                # 应用 RAG Gate
                if apply_rag_gate and weight > 0:
                    logger.info(
                        f"[{self.name}] 开始 RAG Gate 评估: {stock_code}, "
                        f"score={score:.4f}, weight={weight:.2%}"
                    )
                    gate_result = agent._evaluate_gate(
                        stock_code, 
                        engine, 
                        engine.current_date,
                        score,
                        weight
                    )
                    
                    if not gate_result.get("should_execute", True):
                        # RAG 建议不执行交易，设置 weight=0
                        logger.info(
                            f"[{self.name}] RAG Gate 拒绝交易: {stock_code}, "
                            f"score={score:.4f}, weight={weight:.2%}, "
                            f"reasoning={gate_result.get('reasoning', 'N/A')[:100]}"
                        )
                        weight = 0.0
                        agent.gate_skipped_count += 1
                    else:
                        logger.info(
                            f"[{self.name}] RAG Gate 通过交易: {stock_code}, "
                            f"score={score:.4f}, weight={weight:.2%}, "
                            f"reasoning={gate_result.get('reasoning', 'N/A')[:100]}"
                        )
                        agent.gate_passed_count += 1
                elif apply_rag_gate and weight == 0:
                    logger.debug(
                        f"[{self.name}] 跳过 RAG Gate 评估: {stock_code}, "
                        f"weight=0 (无需评估)"
                    )
                
                raw_weights[stock_code] = weight
        else:
            # 并行计算权重
            raw_weights = {}
            
            def _compute_weight_with_gate(stock_code: str, score: float) -> tuple[str, float]:
                """计算单个股票的权重（带 RAG gate）"""
                try:
                    agent = self.agents[stock_code]
                    weight = agent.weight(stock_code, score, engine)
                    
                    # 应用 RAG Gate
                    if apply_rag_gate and weight > 0:
                        logger.info(
                            f"[{self.name}] 开始 RAG Gate 评估: {stock_code}, "
                            f"score={score:.4f}, weight={weight:.2%}"
                        )
                        gate_result = agent._evaluate_gate(
                            stock_code,
                            engine,
                            engine.current_date,
                            score,
                            weight
                        )
                        
                        if not gate_result.get("should_execute", True):
                            # RAG 建议不执行交易，设置 weight=0
                            logger.info(
                                f"[{self.name}] RAG Gate 拒绝交易: {stock_code}, "
                                f"score={score:.4f}, weight={weight:.2%}, "
                                f"reasoning={gate_result.get('reasoning', 'N/A')[:100]}"
                            )
                            weight = 0.0
                            agent.gate_skipped_count += 1
                        else:
                            logger.info(
                                f"[{self.name}] RAG Gate 通过交易: {stock_code}, "
                                f"score={score:.4f}, weight={weight:.2%}, "
                                f"reasoning={gate_result.get('reasoning', 'N/A')[:100]}"
                            )
                            agent.gate_passed_count += 1
                    elif apply_rag_gate and weight == 0:
                        logger.debug(
                            f"[{self.name}] 跳过 RAG Gate 评估: {stock_code}, "
                            f"weight=0 (无需评估)"
                        )
                    
                    return (stock_code, weight)
                except Exception as e:
                    logger.error(f"计算 {stock_code} 的 weight 时出错: {e}", exc_info=True)
                    return (stock_code, 0.0)
            
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                # 提交所有任务
                future_to_stock = {
                    executor.submit(_compute_weight_with_gate, stock_code, score): stock_code
                    for stock_code, score in scores.items()
                }
                
                # 收集结果
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
                # 提交所有任务
                futures = [
                    executor.submit(_update_agent, agent)
                    for agent in self.agents.values()
                ]
                
                # 等待所有任务完成
                for future in as_completed(futures):
                    future.result()  # 检查是否有异常
    
    def get_gate_stats(self) -> Dict[str, Dict[str, int]]:
        """
        获取所有 agent 的 Gate 统计信息
        
        Returns:
            Dict[str, Dict[str, int]]: {stock_code: {gate_passed_count, gate_skipped_count, total_evaluations}}
        """
        stats = {}
        for stock_code, agent in self.agents.items():
            stats[stock_code] = agent.get_gate_stats()
        return stats

