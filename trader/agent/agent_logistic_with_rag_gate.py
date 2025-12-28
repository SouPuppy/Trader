"""
带 RAG Gate 的 Logistic Regression 逻辑回归策略 Agent
使用逻辑回归模型预测股票未来收益，但在执行交易前通过 RAG 系统评估是否应该执行
RAG 系统会检索历史数据、新闻、趋势等信息，基于这些证据进行风险控制决策
"""
import json
import re
import pandas as pd
from typing import Dict, Optional
from datetime import datetime
from trader.agent.agent_logistic import LogisticAgent
from trader.backtest.engine import BacktestEngine
from trader.dataloader import dataloader_ffill
from trader.logger import get_logger

logger = get_logger(__name__)


class LogisticAgentWithRAGGate(LogisticAgent):
    """
    带 RAG Gate 的逻辑回归策略 Agent
    
    继承自 LogisticAgent，添加 RAG Gate 功能：
    1. 使用逻辑回归模型计算 score 和 weight
    2. 在执行交易前，使用 RAG 系统检索相关历史数据、新闻、趋势
    3. 基于 RAG 检索的证据评估是否应该执行交易
    4. 如果 RAG 返回 should_execute=false，跳过本次交易
    """
    
    def __init__(self, name: str = "LogisticAgentWithRAGGate",
                 feature_names: Optional[list] = None,
                 train_window_days: int = 252,
                 prediction_horizon: int = 5,
                 ret_threshold: float = 0.0,
                 retrain_frequency: int = 20,
                 max_position_weight: float = 0.5,
                 min_score_threshold: float = 0.0,
                 max_total_weight: float = 1.0,
                 train_test_split_ratio: float = 0.7,
                 llm_model: str = "deepseek-chat",
                 llm_temperature: float = 0.3,
                 few_shot_count: int = 0,
                 test_mode: bool = False,
                 test_force_reject: bool = False):
        """
        初始化带 RAG Gate 的逻辑回归 Agent
        
        Args:
            name: Agent 名称
            feature_names: 使用的特征名称列表
            train_window_days: 训练窗口大小（交易日数）
            prediction_horizon: 预测未来多少天的收益
            ret_threshold: 收益阈值
            retrain_frequency: 重新训练频率（每N个交易日）
            max_position_weight: 单个股票最大配置比例
            min_score_threshold: 最小 score 阈值
            max_total_weight: 总配置比例上限
            train_test_split_ratio: 训练/测试分割比例
            llm_model: LLM 模型名称（RAG 系统内部使用）
            llm_temperature: LLM 温度参数（控制随机性）
            test_mode: 测试模式，如果为 True，会打印更详细的调试信息
            test_force_reject: 测试模式下的强制拒绝，如果为 True，所有 gate 评估都会返回 False
        """
        super().__init__(
            name=name,
            feature_names=feature_names,
            train_window_days=train_window_days,
            prediction_horizon=prediction_horizon,
            ret_threshold=ret_threshold,
            retrain_frequency=retrain_frequency,
            max_position_weight=max_position_weight,
            min_score_threshold=min_score_threshold,
            max_total_weight=max_total_weight,
            train_test_split_ratio=train_test_split_ratio
        )
        
        self.llm_model = llm_model
        self.llm_temperature = llm_temperature
        self.few_shot_count = few_shot_count
        self.test_mode = test_mode
        self.test_force_reject = test_force_reject
        self.gate_skipped_count = 0  # 记录被 gate 跳过的次数
        self.gate_passed_count = 0  # 记录通过 gate 的次数
    
    def _call_rag(self, question: str, stock_code: str, decision_time: str) -> Optional[Dict]:
        """
        调用 RAG 系统获取答案
        
        Args:
            question: 问题文本
            stock_code: 股票代码
            decision_time: 决策时间（ISO8601 格式）
            
        Returns:
            Dict: RAG 响应，包含 answer, passed, violations 等
            如果调用失败返回 None
        """
        try:
            from trader.rag.answer import rag_answer
        except ImportError:
            logger.error("无法导入 RAG 模块，请确保 RAG 系统已正确配置")
            return None
        
        try:
            logger.info(f"[{self.name}] 调用 RAG 系统: {stock_code} on {decision_time}")
            
            # 调用 RAG 系统
            verified_answer = rag_answer(
                question=question,
                stock_code=stock_code,
                decision_time=decision_time,
                frequency="1d",
                few_shot_count=self.few_shot_count
            )
            
            if verified_answer:
                logger.info(f"[{self.name}] RAG 系统调用成功: passed={verified_answer.passed}, mode={verified_answer.mode}")
                return {
                    "answer": verified_answer.answer,
                    "passed": verified_answer.passed,
                    "violations": verified_answer.violations,
                    "mode": verified_answer.mode
                }
            else:
                logger.warning(f"[{self.name}] RAG 系统返回空结果")
                return None
                
        except Exception as e:
            logger.error(f"调用 RAG 系统时出错: {e}", exc_info=True)
            return None
    
    def _parse_rag_answer(self, rag_response: Dict, default_execute: bool = True) -> Dict[str, any]:
        """
        解析 RAG 返回的答案，提取 should_execute 决策
        
        Args:
            rag_response: RAG 系统返回的响应
            default_execute: 如果无法解析时的默认值
            
        Returns:
            Dict: 包含 should_execute, reasoning 的字典
        """
        answer_text = rag_response.get("answer", "")
        passed = rag_response.get("passed", False)
        violations = rag_response.get("violations", [])
        mode = rag_response.get("mode", "normal")
        
        # 如果 RAG 验证失败或处于降级模式，倾向于拒绝执行
        if not passed or mode == "degraded":
            logger.warning(f"[{self.name}] RAG 验证失败或降级模式，倾向于拒绝执行")
            return {
                "should_execute": False,
                "reasoning": f"RAG 验证失败 (passed={passed}, mode={mode}). 违反项: {violations}"
            }
        
        # 尝试从答案中提取 JSON 格式的决策
        # 首先尝试直接解析 JSON
        json_match = re.search(r'\{[^{}]*"should_execute"[^{}]*\}', answer_text, re.DOTALL)
        if json_match:
            try:
                json_str = json_match.group(0)
                result = json.loads(json_str)
                if 'should_execute' in result:
                    should_execute = result.get('should_execute', default_execute)
                    reasoning = result.get('reasoning', answer_text[:200])
                    
                    # 转换为布尔值
                    if isinstance(should_execute, bool):
                        pass
                    elif isinstance(should_execute, str):
                        should_execute = should_execute.lower() in ("true", "1", "yes", "y")
                    else:
                        should_execute = default_execute
                    
                    return {
                        "should_execute": should_execute,
                        "reasoning": reasoning
                    }
            except json.JSONDecodeError:
                pass
        
        # 如果无法解析 JSON，尝试从文本中提取决策
        # 查找明确的拒绝信号
        reject_keywords = [
            "should not execute", "do not execute", "should not trade",
            "skip this trade", "avoid trading", "should_execute: false",
            "should_execute=false", "should_execute:false"
        ]
        
        accept_keywords = [
            "should execute", "can execute", "should trade",
            "proceed with trade", "should_execute: true",
            "should_execute=true", "should_execute:true"
        ]
        
        answer_lower = answer_text.lower()
        
        # 检查拒绝关键词
        for keyword in reject_keywords:
            if keyword in answer_lower:
                return {
                    "should_execute": False,
                    "reasoning": answer_text[:300]
                }
        
        # 检查接受关键词
        for keyword in accept_keywords:
            if keyword in answer_lower:
                return {
                    "should_execute": True,
                    "reasoning": answer_text[:300]
                }
        
        # 如果答案中包含明确的负面词汇，倾向于拒绝
        negative_keywords = [
            "extreme volatility", "market crash", "panic selling",
            "major scandal", "bankruptcy risk", "financial crisis",
            "regulatory changes", "severe negative", "critical risk"
        ]
        
        positive_keywords = [
            "normal conditions", "moderate volatility", "reasonable market",
            "stable trend", "positive signal", "favorable conditions"
        ]
        
        negative_count = sum(1 for kw in negative_keywords if kw in answer_lower)
        positive_count = sum(1 for kw in positive_keywords if kw in answer_lower)
        
        if negative_count > positive_count:
            return {
                "should_execute": False,
                "reasoning": f"检测到负面信号 (negative: {negative_count}, positive: {positive_count}). {answer_text[:200]}"
            }
        elif positive_count > negative_count:
            return {
                "should_execute": True,
                "reasoning": f"检测到正面信号 (positive: {positive_count}, negative: {negative_count}). {answer_text[:200]}"
            }
        
        # 默认值：如果无法确定，使用 default_execute
        logger.warning(f"[{self.name}] 无法从 RAG 答案中提取明确决策，使用默认值: {default_execute}")
        return {
            "should_execute": default_execute,
            "reasoning": f"无法明确解析决策，使用默认值。RAG 答案: {answer_text[:200]}"
        }
    
    def _evaluate_gate(self, stock_code: str, engine: BacktestEngine, 
                       date: str, score: float, target_weight: float) -> Dict[str, any]:
        """
        通过 RAG 系统评估是否应该执行交易（gate decision）
        
        Args:
            stock_code: 股票代码
            engine: 回测引擎
            date: 当前日期
            score: 模型预测的 score
            target_weight: 目标权重
            
        Returns:
            Dict: 包含 should_execute, reasoning 的字典
        """
        # 获取账户信息
        current_position = engine.account.get_position(stock_code)
        position_str = "None"
        if current_position:
            position_str = f"{current_position['shares']} shares @ {current_position['average_price']:.2f}"
        
        # 获取账户权益
        market_prices = engine.get_market_prices([stock_code])
        account_equity = engine.account.equity(market_prices)
        
        # 构建 RAG 问题
        # 使用 risk_check 任务类型，让 RAG 系统检索风险相关信息
        question = f"""Based on recent market data, news, and trends for {stock_code}, should I execute a trade on {date}?

Context:
- Logistic regression model score: {score:.4f} (positive = buy signal, negative = sell/reduce signal)
- Target position weight: {target_weight:.2%}
- Current position: {position_str}
- Account equity: {account_equity:.2f}
- Current cash: {engine.account.cash:.2f}

Please analyze the following risks and provide a decision:
1. Market volatility and extreme conditions
2. Recent news impact and sentiment
3. Technical indicators and trend signals
4. Systemic risks or major events
5. Model signal alignment with market conditions

Please provide your decision in JSON format:
{{
    "should_execute": true_or_false (boolean),
    "reasoning": "Brief explanation based on evidence (max 200 words)"
}}

Decision Guidelines:
- should_execute = TRUE (default) if:
  * Model signal aligns with reasonable market conditions
  * Moderate volatility is acceptable
  * No extreme market events detected
  * Normal trading conditions
  
- should_execute = FALSE (skip trade in extreme cases) if:
  * EXTREME volatility (e.g., market crash, panic selling)
  * SEVERE negative sentiment (e.g., major scandal, bankruptcy risk)
  * CRITICAL technical indicators (e.g., major trend reversal, support break)
  * EXTREME overvaluation/undervaluation (e.g., bubble conditions)
  * MAJOR systemic risks (e.g., financial crisis, regulatory changes)
  * Model signal contradicts strong market signals

Remember: The logistic regression model provides a data-driven signal. Only skip if conditions are TRULY EXTREME and would significantly harm the trade. Normal market fluctuations and moderate volatility should NOT prevent execution."""

        # 转换日期格式为 ISO8601
        try:
            date_dt = pd.to_datetime(date)
            # RAG 系统需要 ISO8601 格式，使用 00:00:00 作为时间部分
            decision_time = date_dt.strftime("%Y-%m-%dT00:00:00")
        except Exception as e:
            logger.warning(f"无法解析日期 {date}，尝试直接使用: {e}")
            # 如果日期格式已经是 YYYY-MM-DD，添加时间部分
            if len(date) == 10 and date.count('-') == 2:
                decision_time = f"{date}T00:00:00"
            else:
                decision_time = date
        
        # 测试模式：强制拒绝
        if self.test_force_reject:
            logger.warning(f"[{self.name}] 测试模式：强制拒绝本次交易")
            return {
                "should_execute": False,
                "reasoning": "测试模式：强制拒绝"
            }
        
        # 调用 RAG 系统
        logger.info(f"[{self.name}] 调用 RAG Gate 评估: {stock_code} on {date}, score={score:.4f}, weight={target_weight:.2%}")
        
        if self.test_mode:
            logger.info(f"[{self.name}] 测试模式：显示完整问题（前500字符）: {question[:500]}...")
        
        rag_response = self._call_rag(question, stock_code, decision_time)
        
        if not rag_response:
            logger.warning(f"[{self.name}] RAG 调用失败，默认执行交易")
            return {
                "should_execute": True,
                "reasoning": "RAG 调用失败，默认执行"
            }
        
        if self.test_mode:
            logger.info(f"[{self.name}] 测试模式：RAG 原始响应: {rag_response}")
        
        # 解析 RAG 答案
        result = self._parse_rag_answer(rag_response, default_execute=True)
        
        should_execute = result.get("should_execute", True)
        reasoning = result.get("reasoning", "No reasoning provided")
        
        logger.info(
            f"[{self.name}] RAG Gate 评估结果: {stock_code} on {date}, "
            f"score={score:.4f}, weight={target_weight:.2%}, "
            f"should_execute={should_execute}, reasoning={reasoning[:100]}..."
        )
        
        return {
            "should_execute": should_execute,
            "reasoning": reasoning
        }
    
    def get_gate_stats(self) -> Dict[str, int]:
        """
        获取 Gate 统计信息
        
        Returns:
            Dict: 包含 gate_passed_count, gate_skipped_count 的字典
        """
        return {
            "gate_passed_count": self.gate_passed_count,
            "gate_skipped_count": self.gate_skipped_count,
            "total_evaluations": self.gate_passed_count + self.gate_skipped_count
        }

