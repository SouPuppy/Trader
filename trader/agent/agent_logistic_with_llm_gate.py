"""
带 LLM Gate 的 Logistic Regression 逻辑回归策略 Agent
使用逻辑回归模型预测股票未来收益，但在执行交易前通过 LLM 评估是否应该执行
"""
import json
import re
import pandas as pd
from typing import Dict, Optional
from trader.agent.agent_logistic import LogisticAgent
from trader.backtest.engine import BacktestEngine
from trader.dataloader import dataloader_ffill
from trader.logger import get_logger

logger = get_logger(__name__)

# LLM Prompt Template for Logistic Strategy Gate Decision
LLM_LOGISTIC_GATE_PROMPT_TEMPLATE = """
You are a professional stock market analyst. Analyze the following stock features and market data to decide whether to execute a trade based on a logistic regression model prediction.

IMPORTANT CONTEXT: The logistic regression model has predicted a trading signal (score: {score:.4f}). A positive score suggests buying, while a negative score suggests selling or reducing position. However, we want to avoid trading during EXTREME market conditions that could significantly harm returns.

Stock Symbol: {symbol}
Current Date: {date}
Model Score: {score:.4f} (positive = buy signal, negative = sell/reduce signal)
Target Weight: {target_weight:.2%} (desired position size)

Features Data:
{features_str}

Account Information:
- Current Cash: {cash:.2f}
- Current Position: {current_position}
- Account Equity: {equity:.2f}

Please analyze the market conditions and provide your decision in JSON format:
{{
    "should_execute": true_or_false (boolean),
    "reasoning": "Brief explanation of your decision (max 200 words)"
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

Remember: The logistic regression model provides a data-driven signal. Only skip if conditions are TRULY EXTREME and would significantly harm the trade. Normal market fluctuations and moderate volatility should NOT prevent execution.
"""


class LogisticAgentWithLLMGate(LogisticAgent):
    """
    带 LLM Gate 的逻辑回归策略 Agent
    
    继承自 LogisticAgent，添加 LLM Gate 功能：
    1. 使用逻辑回归模型计算 score 和 weight
    2. 在执行交易前，使用 LLM 评估是否应该执行
    3. 如果 LLM 返回 should_execute=false，跳过本次交易
    """
    
    def __init__(self, name: str = "LogisticAgentWithLLMGate",
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
                 test_mode: bool = False,
                 test_force_reject: bool = False):
        """
        初始化带 LLM Gate 的逻辑回归 Agent
        
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
            llm_model: LLM 模型名称
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
        self.test_mode = test_mode
        self.test_force_reject = test_force_reject
        self.gate_skipped_count = 0  # 记录被 gate 跳过的次数
        self.gate_passed_count = 0  # 记录通过 gate 的次数
    
    def _call_llm(self, prompt: str) -> Optional[Dict]:
        """
        调用 LLM API
        
        Args:
            prompt: 提示词
            
        Returns:
            Dict: LLM 响应，包含 should_execute, reasoning
            如果调用失败返回 None
        """
        try:
            from openai import OpenAI
        except ImportError:
            logger.error("未安装 openai 模块，无法调用 LLM。请运行: pip install openai")
            return None
        
        try:
            from trader.config import get_deepseek_api_key
            api_key = get_deepseek_api_key()
            client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")
            
            response = client.chat.completions.create(
                model=self.llm_model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a professional stock market analyst specializing in quantitative trading strategies. Provide clear, concise decisions based on market analysis."
                    },
                    {"role": "user", "content": prompt}
                ],
                stream=False,
                temperature=self.llm_temperature
            )
            
            content = response.choices[0].message.content
            
            # 解析 JSON 响应
            json_str = content.strip()
            json_str = re.sub(r'```json\s*', '', json_str)
            json_str = re.sub(r'```\s*', '', json_str)
            json_str = json_str.strip()
            
            # 尝试提取 JSON 对象
            result = None
            try:
                result = json.loads(json_str)
            except json.JSONDecodeError:
                start_idx = json_str.find('{')
                end_idx = json_str.rfind('}')
                if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
                    json_str = json_str[start_idx:end_idx + 1]
                    try:
                        result = json.loads(json_str)
                    except json.JSONDecodeError:
                        logger.warning(f"无法解析 LLM 响应为 JSON: {json_str[:200]}")
                        return None
            
            # 验证响应格式
            if result and isinstance(result, dict):
                if 'should_execute' in result:
                    return result
            
            logger.warning(f"LLM 响应格式不正确: {result}")
            return None
            
        except Exception as e:
            logger.error(f"调用 LLM API 时出错: {e}", exc_info=True)
            return None
    
    def _evaluate_gate(self, stock_code: str, engine: BacktestEngine, 
                       date: str, score: float, target_weight: float) -> Dict[str, any]:
        """
        通过 LLM 评估是否应该执行交易（gate decision）
        
        Args:
            stock_code: 股票代码
            engine: 回测引擎
            date: 当前日期
            score: 模型预测的 score
            target_weight: 目标权重
            
        Returns:
            Dict: 包含 should_execute, reasoning 的字典
        """
        # 使用 dataloader 获取所有 features
        try:
            df = engine.get_features_with_dataloader(
                stock_code,
                start_date=date,  # 从当前日期开始
                end_date=date,
                dataloader=dataloader_ffill(stock_code),
                force=False
            )
            
            if df.empty:
                logger.warning(f"无法获取 {stock_code} 在 {date} 的特征数据，默认执行交易")
                return {
                    "should_execute": True,
                    "reasoning": "无法获取特征数据，默认执行"
                }
            
            # 获取当前日期的特征值
            date_dt = pd.to_datetime(date)
            if date_dt in df.index:
                features_row = df.loc[date_dt]
            else:
                available_rows = df[df.index <= date_dt]
                if available_rows.empty:
                    logger.warning(f"未找到 {stock_code} 在 {date} 或之前的数据，默认执行交易")
                    return {
                        "should_execute": True,
                        "reasoning": "未找到历史数据，默认执行"
                    }
                features_row = available_rows.iloc[-1]
            
            # 转换为字典，过滤掉 None 值
            features_dict = {}
            for col in df.columns:
                value = features_row[col]
                if pd.notna(value):
                    features_dict[col] = float(value)
            
            # 格式化 features 为字符串
            features_str = "\n".join([
                f"  {name}: {value:.6f}" if isinstance(value, float) else f"  {name}: {value}"
                for name, value in features_dict.items()
            ])
            
        except Exception as e:
            logger.error(f"获取特征数据时出错: {e}", exc_info=True)
            return {
                "should_execute": True,
                "reasoning": f"获取特征数据失败: {e}，默认执行"
            }
        
        # 获取账户信息
        current_position = engine.account.get_position(stock_code)
        position_str = "None"
        if current_position:
            position_str = f"{current_position['shares']} shares @ {current_position['average_price']:.2f}"
        
        # 获取账户权益
        market_prices = engine.get_market_prices([stock_code])
        account_equity = engine.account.equity(market_prices)
        
        # 构建 prompt
        prompt = LLM_LOGISTIC_GATE_PROMPT_TEMPLATE.format(
            symbol=stock_code,
            date=date,
            score=score,
            target_weight=target_weight,
            features_str=features_str,
            cash=engine.account.cash,
            equity=account_equity,
            current_position=position_str
        )
        
        # 测试模式：强制拒绝
        if self.test_force_reject:
            logger.warning(f"[{self.name}] 测试模式：强制拒绝本次交易")
            return {
                "should_execute": False,
                "reasoning": "测试模式：强制拒绝"
            }
        
        # 调用 LLM
        logger.info(f"[{self.name}] 调用 LLM Gate 评估: {stock_code} on {date}, score={score:.4f}, weight={target_weight:.2%}")
        
        if self.test_mode:
            logger.info(f"[{self.name}] 测试模式：显示完整 prompt（前500字符）: {prompt[:500]}...")
        
        response = self._call_llm(prompt)
        
        if response:
            logger.info(f"[{self.name}] LLM API 调用成功: {stock_code} on {date}")
        else:
            logger.warning(f"[{self.name}] LLM API 调用失败: {stock_code} on {date}")
        
        if not response:
            logger.warning(f"[{self.name}] LLM 调用失败，默认执行交易")
            return {
                "should_execute": True,
                "reasoning": "LLM 调用失败，默认执行"
            }
        
        if self.test_mode:
            logger.info(f"[{self.name}] 测试模式：LLM 原始响应: {response}")
        
        # 处理 should_execute 字段（可能是布尔值或字符串）
        should_execute_raw = response.get("should_execute", True)
        
        # 转换为布尔值（处理字符串 "true"/"false" 的情况）
        if isinstance(should_execute_raw, bool):
            should_execute = should_execute_raw
        elif isinstance(should_execute_raw, str):
            should_execute = should_execute_raw.lower() in ("true", "1", "yes", "y")
        else:
            # 默认值：如果无法解析，使用 False（保守策略）
            logger.warning(f"[{self.name}] 无法解析 should_execute 值: {should_execute_raw}，使用 False（保守策略）")
            should_execute = False
        
        reasoning = response.get("reasoning", "No reasoning provided")
        
        logger.info(
            f"[{self.name}] LLM Gate 评估结果: {stock_code} on {date}, "
            f"score={score:.4f}, weight={target_weight:.2%}, "
            f"should_execute={should_execute} (原始值: {should_execute_raw}), reasoning={reasoning[:100]}..."
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

