"""
带 LLM Gate 的 DCA (Dollar Cost Averaging) 定投策略 Agent
每月固定金额买入指定股票，但在执行前通过 LLM 评估是否应该执行
"""
import json
import re
import pandas as pd
from typing import Dict, List, Optional
from datetime import datetime
from trader.agent.agent_dca import DCAAgent
from trader.backtest.engine import BacktestEngine
from trader.dataloader import dataloader_ffill
from trader.logger import get_logger

logger = get_logger(__name__)

# LLM Prompt Template for Gate Decision
LLM_GATE_PROMPT_TEMPLATE = """
You are a professional stock market analyst. Analyze the following stock features and market data to decide whether to execute a Dollar Cost Averaging (DCA) investment at this time.

IMPORTANT CONTEXT: DCA (Dollar Cost Averaging) is a strategy that involves investing a fixed amount regularly, regardless of market conditions. The goal is to average out the purchase price over time. However, we want to avoid investing during EXTREME market conditions that could significantly harm returns.

Stock Symbol: {symbol}
Current Date: {date}

Features Data:
{features_str}

Account Information:
- Current Cash: {cash:.2f}
- Monthly Investment Amount: {monthly_investment:.2f}
- Current Position: {current_position}

Please analyze the market conditions and provide your decision in JSON format:
{{
    "should_execute": true_or_false (boolean),
    "reasoning": "Brief explanation of your decision (max 200 words)"
}}

Decision Guidelines:
- should_execute = TRUE (default for DCA) if:
  * Normal market conditions (even if not perfect)
  * Moderate volatility is acceptable
  * Neutral to slightly negative sentiment is acceptable
  * Slightly overvalued conditions are acceptable (DCA averages out)
  * Regular market fluctuations are expected and acceptable
  
- should_execute = FALSE (only skip in extreme cases) if:
  * EXTREME volatility (e.g., market crash, panic selling)
  * SEVERE negative sentiment (e.g., major scandal, bankruptcy risk)
  * CRITICAL technical indicators (e.g., major trend reversal, support break)
  * EXTREME overvaluation (e.g., bubble conditions, unsustainable P/E > 50)
  * MAJOR systemic risks (e.g., financial crisis, regulatory changes)

Remember: DCA works best when executed consistently. Only skip if conditions are TRULY EXTREME and would significantly harm the investment. Normal market fluctuations, moderate volatility, and typical valuation levels should NOT prevent DCA execution.
"""


class DCAAgentWithLLMGate(DCAAgent):
    """
    带 LLM Gate 的 DCA 策略 Agent
    
    继承自 DCAAgent，添加 LLM Gate 功能：
    1. 使用 dataloader 获取所有 features
    2. 将 features 和 prompt template 传给 LLM
    3. LLM 评估是否应该执行本次定投（gate decision）
    4. 如果 LLM 返回 should_execute=false，跳过本次定投
    """
    
    def __init__(self, name: str = "DCAAgentWithLLMGate",
                 monthly_investment: float = 1000.0,
                 dca_frequency: str = "monthly",
                 llm_model: str = "deepseek-chat",
                 llm_temperature: float = 0.3,
                 test_mode: bool = False,
                 test_force_reject: bool = False):
        """
        初始化带 LLM Gate 的 DCA Agent
        
        Args:
            name: Agent 名称
            monthly_investment: 每月定投金额（元）
            dca_frequency: 定投频率，"monthly"（每月）或 "daily"（每日）
            llm_model: LLM 模型名称
            llm_temperature: LLM 温度参数（控制随机性）
            test_mode: 测试模式，如果为 True，会打印更详细的调试信息
            test_force_reject: 测试模式下的强制拒绝，如果为 True，所有 gate 评估都会返回 False
        """
        super().__init__(name, monthly_investment, dca_frequency)
        
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
                        "content": "You are a professional stock market analyst specializing in Dollar Cost Averaging strategies. Provide clear, concise decisions based on market analysis."
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
                       date: str) -> Dict[str, any]:
        """
        通过 LLM 评估是否应该执行定投（gate decision）
        
        Args:
            stock_code: 股票代码
            engine: 回测引擎
            date: 当前日期
            
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
                logger.warning(f"无法获取 {stock_code} 在 {date} 的特征数据，默认执行定投")
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
                    logger.warning(f"未找到 {stock_code} 在 {date} 或之前的数据，默认执行定投")
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
        
        # 构建 prompt
        prompt = LLM_GATE_PROMPT_TEMPLATE.format(
            symbol=stock_code,
            date=date,
            features_str=features_str,
            cash=engine.account.cash,
            monthly_investment=self.monthly_investment,
            current_position=position_str
        )
        
        # 测试模式：强制拒绝
        if self.test_force_reject:
            logger.warning(f"[{self.name}] 测试模式：强制拒绝本次定投")
            return {
                "should_execute": False,
                "reasoning": "测试模式：强制拒绝"
            }
        
        # 调用 LLM
        logger.debug(f"[{self.name}] 调用 LLM Gate 评估: {stock_code} on {date}")
        
        if self.test_mode:
            logger.info(f"[{self.name}] 测试模式：显示完整 prompt（前500字符）: {prompt[:500]}...")
        
        response = self._call_llm(prompt)
        
        if not response:
            logger.warning(f"[{self.name}] LLM 调用失败，默认执行定投")
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
            f"should_execute={should_execute} (原始值: {should_execute_raw}), reasoning={reasoning[:100]}..."
        )
        
        return {
            "should_execute": should_execute,
            "reasoning": reasoning
        }
    
    def on_date(self, engine: BacktestEngine, date: str):
        """
        每个交易日的回调函数
        
        实现带 LLM Gate 的定投策略：
        1. 判断是否应该定投（时间条件）
        2. 使用 LLM Gate 评估是否应该执行
        3. 如果通过 gate，执行定投；否则跳过
        
        Args:
            engine: 回测引擎
            date: 当前日期
        """
        if not self.should_dca_today(date):
            return
        
        if not self.dca_stock_codes:
            return
        
        try:
            # 计算每个股票的定投金额（如果多个股票，平均分配）
            num_stocks = len(self.dca_stock_codes)
            investment_per_stock = self.monthly_investment / num_stocks
            
            for stock_code in self.dca_stock_codes:
                try:
                    # 获取当前价格
                    price = engine.get_current_price(stock_code)
                    if price is None:
                        logger.warning(f"[{date}] 无法获取 {stock_code} 的价格，跳过定投")
                        continue
                    
                    # 检查可用现金是否足够
                    if engine.account.cash < investment_per_stock:
                        logger.warning(
                            f"[{date}] 现金不足，无法定投 {stock_code}: "
                            f"需要 {investment_per_stock:.2f} 元，当前现金 {engine.account.cash:.2f} 元"
                        )
                        continue
                    
                    # 使用 LLM Gate 评估是否应该执行
                    logger.info(f"[{self.name}] 开始 LLM Gate 评估: {stock_code} on {date}")
                    gate_result = self._evaluate_gate(stock_code, engine, date)
                    
                    should_execute = gate_result.get("should_execute", True)
                    reasoning = gate_result.get("reasoning", "")
                    
                    if not should_execute:
                        # Gate 拒绝执行，跳过本次定投
                        self.gate_skipped_count += 1
                        logger.info(
                            f"[{self.name}] LLM Gate 拒绝执行定投: {stock_code} on {date}, "
                            f"原因: {reasoning[:100]}..."
                        )
                        continue
                    
                    # Gate 通过，执行定投
                    self.gate_passed_count += 1
                    logger.info(
                        f"[{self.name}] LLM Gate 通过，执行定投: {stock_code} on {date}, "
                        f"原因: {reasoning[:100]}..."
                    )
                    
                    # 执行定投买入（使用固定金额）
                    engine.buy(stock_code, amount=investment_per_stock)
                    self.investment_count += 1
                    logger.info(
                        f"[{date}] 定投买入 {stock_code}: {investment_per_stock:.2f} 元 @ {price:.2f}"
                    )
                    
                except Exception as e:
                    logger.error(f"[{date}] 定投 {stock_code} 时出错: {e}", exc_info=True)
        except Exception as e:
            logger.error(f"[{date}] 执行定投策略时出错: {e}", exc_info=True)
    
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

