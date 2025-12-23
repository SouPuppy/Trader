"""
基于 LLM 的 DCA 策略 Agent（带不确定性风险控制）
每月固定金额买入指定股票，但通过 LLM 评估不确定性，并使用 UncertaintyGateRiskManager 进行风险控制
"""
import json
import re
import pandas as pd
from typing import Dict, List, Optional
from datetime import datetime
from trader.agent.TradingAgent import TradingAgent
from trader.agent.agent_dca import DCAAgent
from trader.backtest.engine import BacktestEngine
from trader.risk.OrderIntent import OrderIntent, OrderSide, PriceType
from trader.risk.RiskManager import RiskManagerPipeline, RiskContext
from trader.risk.control_leverage_limit import LeverageLimitRiskManager
from trader.risk.control_uncertainty_gate import UncertaintyGateRiskManager
from trader.dataloader import dataloader_ffill
from trader.logger import get_logger

logger = get_logger(__name__)

# LLM Prompt Template
LLM_PROMPT_TEMPLATE = """
You are a professional stock market analyst. Analyze the following stock features and market data to assess the investment decision for a Dollar Cost Averaging (DCA) strategy.

Stock Symbol: {symbol}
Current Date: {date}

Features Data:
{features_str}

Account Information:
- Current Cash: {cash:.2f}
- Monthly Investment Amount: {monthly_investment:.2f}
- Current Position: {current_position}

Please analyze the market conditions and provide your assessment in JSON format:
{{
    "sentiment": sentiment_value (float from -1.0 to 1.0, where 1.0 is very bullish, -1.0 is very bearish),
    "confidence": confidence_value (float from 0.0 to 1.0, where 1.0 is very confident, 0.0 is very uncertain),
    "reasoning": "Brief explanation of your assessment (max 200 words)"
}}

Guidelines:
- sentiment: Your view on whether this is a good time to buy (1.0 = excellent time, 0.0 = neutral, -1.0 = bad time)
- confidence: How confident you are in your assessment (1.0 = very confident, 0.0 = very uncertain)
- reasoning: Explain the key factors influencing your decision
"""


class LLMDCAAgentWithRiskControl(DCAAgent):
    """
    基于 LLM 的 DCA 策略 Agent（带不确定性风险控制）
    
    继承自 DCAAgent，添加 LLM 不确定性评估和风险控制功能：
    1. 使用 dataloader 获取所有 features
    2. 将 features 和 prompt template 传给 LLM
    3. 多次调用 LLM 来评估不确定性（通过多次调用并计算置信度）
    4. 使用 UncertaintyGateRiskManager 进行风险控制
    """
    
    def __init__(self, name: str = "LLMDCAAgentWithRiskControl",
                 monthly_investment: float = 1000.0,
                 dca_frequency: str = "monthly",
                 max_leverage: float = 1.0,
                 llm_model: str = "deepseek-chat",
                 llm_temperature: float = 0.3,
                 llm_num_calls: int = 3,
                 uncertainty_high_threshold: float = 0.7,
                 uncertainty_medium_threshold: float = 0.4,
                 uncertainty_scale_down_factor: float = 0.5):
        """
        初始化基于 LLM 的 DCA Agent
        
        Args:
            name: Agent 名称
            monthly_investment: 每月定投金额（元）
            dca_frequency: 定投频率，"monthly"（每月）或 "daily"（每日）
            max_leverage: 最大杠杆率（传递给 LeverageLimitRiskManager）
            llm_model: LLM 模型名称
            llm_temperature: LLM 温度参数（控制随机性）
            llm_num_calls: LLM 调用次数（用于评估不确定性）
            uncertainty_high_threshold: 高置信度阈值
            uncertainty_medium_threshold: 中等置信度阈值
            uncertainty_scale_down_factor: 中等不确定性时的仓位缩放因子
        """
        super().__init__(name, monthly_investment, dca_frequency)
        
        self.llm_model = llm_model
        self.llm_temperature = llm_temperature
        self.llm_num_calls = llm_num_calls
        
        # 创建风险管理器管道
        self.risk_pipeline = RiskManagerPipeline([
            UncertaintyGateRiskManager(
                name="UncertaintyGate",
                high_threshold=uncertainty_high_threshold,
                medium_threshold=uncertainty_medium_threshold,
                scale_down_factor=uncertainty_scale_down_factor
            ),
            LeverageLimitRiskManager(
                name="LeverageLimit",
                max_leverage=max_leverage
            )
        ])
    
    def _call_llm(self, prompt: str) -> Optional[Dict]:
        """
        调用 LLM API
        
        Args:
            prompt: 提示词
            
        Returns:
            Dict: LLM 响应，包含 sentiment, confidence, reasoning
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
                        "content": "You are a professional stock market analyst specializing in Dollar Cost Averaging strategies."
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
                if 'sentiment' in result and 'confidence' in result:
                    return result
            
            logger.warning(f"LLM 响应格式不正确: {result}")
            return None
            
        except Exception as e:
            logger.error(f"调用 LLM API 时出错: {e}", exc_info=True)
            return None
    
    def _evaluate_uncertainty(self, stock_code: str, engine: BacktestEngine, 
                              date: str) -> Dict[str, float]:
        """
        通过多次调用 LLM 评估不确定性
        
        Args:
            stock_code: 股票代码
            engine: 回测引擎
            date: 当前日期
            
        Returns:
            Dict: 包含 sentiment, confidence, reasoning 的字典
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
                logger.warning(f"无法获取 {stock_code} 在 {date} 的特征数据")
                # 返回默认值
                return {
                    "sentiment": 0.0,
                    "confidence": 0.5,
                    "reasoning": "无法获取特征数据"
                }
            
            # 获取当前日期的特征值
            date_dt = pd.to_datetime(date)
            if date_dt in df.index:
                features_row = df.loc[date_dt]
            else:
                available_rows = df[df.index <= date_dt]
                if available_rows.empty:
                    logger.warning(f"未找到 {stock_code} 在 {date} 或之前的数据")
                    return {
                        "sentiment": 0.0,
                        "confidence": 0.5,
                        "reasoning": "未找到历史数据"
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
                "sentiment": 0.0,
                "confidence": 0.5,
                "reasoning": f"获取特征数据失败: {e}"
            }
        
        # 获取账户信息
        current_position = engine.account.get_position(stock_code)
        position_str = "None"
        if current_position:
            position_str = f"{current_position['shares']} shares @ {current_position['average_price']:.2f}"
        
        # 构建 prompt
        prompt = LLM_PROMPT_TEMPLATE.format(
            symbol=stock_code,
            date=date,
            features_str=features_str,
            cash=engine.account.cash,
            monthly_investment=self.monthly_investment,
            current_position=position_str
        )
        
        # 多次调用 LLM 来评估不确定性
        responses = []
        for i in range(self.llm_num_calls):
            logger.debug(f"[{self.name}] LLM 调用 {i+1}/{self.llm_num_calls} for {stock_code} on {date}")
            response = self._call_llm(prompt)
            if response:
                responses.append(response)
        
        if not responses:
            logger.warning(f"[{self.name}] 所有 LLM 调用都失败，使用默认值")
            return {
                "sentiment": 0.0,
                "confidence": 0.3,  # 低置信度，因为调用失败
                "reasoning": "LLM 调用失败"
            }
        
        # 计算平均 sentiment 和 confidence
        sentiments = [r.get("sentiment", 0.0) for r in responses]
        confidences = [r.get("confidence", 0.0) for r in responses]
        
        avg_sentiment = sum(sentiments) / len(sentiments)
        avg_confidence = sum(confidences) / len(confidences)
        
        # 计算置信度的标准差（用于评估不确定性）
        if len(confidences) > 1:
            import statistics
            confidence_std = statistics.stdev(confidences)
            # 标准差越大，不确定性越高，因此调整置信度
            adjusted_confidence = max(0.0, avg_confidence - confidence_std)
        else:
            adjusted_confidence = avg_confidence
        
        # 合并所有 reasoning
        reasoning_list = [r.get("reasoning", "") for r in responses if r.get("reasoning")]
        reasoning = " | ".join(reasoning_list) if reasoning_list else "No reasoning provided"
        
        logger.info(
            f"[{self.name}] LLM 评估结果: {stock_code} on {date}, "
            f"sentiment={avg_sentiment:.2f}, confidence={adjusted_confidence:.2f} "
            f"(原始={avg_confidence:.2f}, 调用次数={len(responses)})"
        )
        
        return {
            "sentiment": avg_sentiment,
            "confidence": adjusted_confidence,
            "reasoning": reasoning
        }
    
    def apply_risk_control(self, order_intents: List[OrderIntent], 
                          engine: BacktestEngine) -> List[OrderIntent]:
        """
        应用风险控制管道
        
        Args:
            order_intents: 订单意图列表
            engine: 回测引擎
            
        Returns:
            List[OrderIntent]: 通过风控后的订单意图列表
        """
        if not order_intents:
            return []
        
        # 创建风险控制上下文
        market_prices = {}
        for order in order_intents:
            if order.symbol not in market_prices:
                price = engine.get_price(order.symbol)
                if price is not None:
                    market_prices[order.symbol] = price
        
        # 获取所有持仓的市场价格
        for symbol in engine.account.positions.keys():
            if symbol not in market_prices:
                price = engine.get_price(symbol)
                if price is not None:
                    market_prices[symbol] = price
        
        ctx = RiskContext(
            account=engine.account,
            market=engine.market,
            current_date=engine.current_date,
            market_prices=market_prices
        )
        
        # 1. 验证订单（过滤掉被拒绝的订单）
        validated_orders = []
        for order in order_intents:
            ok, reason = self.risk_pipeline.validate(ctx, order)
            if ok:
                validated_orders.append(order)
            else:
                logger.warning(f"[{self.name}] 订单被拒绝: {order}, 原因: {reason}")
        
        # 2. 调整订单（削减规模以符合风险限制）
        adjusted_orders = []
        for order in validated_orders:
            adjusted_order = self.risk_pipeline.adjust(ctx, order)
            # 检查调整后的订单是否有效
            if adjusted_order.qty is not None and adjusted_order.qty > 0:
                adjusted_orders.append(adjusted_order)
            elif adjusted_order.target_weight is not None and adjusted_order.target_weight > 0:
                adjusted_orders.append(adjusted_order)
        
        # 3. 组合层面调整
        final_orders = self.risk_pipeline.pre_trade(ctx, adjusted_orders)
        
        return final_orders
    
    def execute_orders(self, order_intents: List[OrderIntent], engine: BacktestEngine):
        """
        执行订单意图（转换为 Engine 的 Action）
        
        Args:
            order_intents: 订单意图列表
            engine: 回测引擎
        """
        for order in order_intents:
            try:
                if order.side == OrderSide.BUY:
                    if order.qty is not None and order.qty > 0:
                        engine.buy(order.symbol, shares=order.qty)
                        dca_amount = order.metadata.get("dca_amount", 0.0)
                        logger.info(
                            f"[{self.name}] 执行买入订单: {order.symbol}, "
                            f"股数 {order.qty}, "
                            f"定投金额 {dca_amount:.2f}, "
                            f"置信度 {order.confidence:.2f}"
                        )
                    elif order.target_weight is not None:
                        market_prices = engine.get_market_prices([order.symbol])
                        if order.symbol not in market_prices:
                            logger.warning(f"无法获取 {order.symbol} 的价格，跳过订单")
                            continue
                        
                        equity = engine.account.equity(market_prices)
                        target_value = equity * order.target_weight
                        
                        current_position = engine.account.get_position(order.symbol)
                        current_value = 0.0
                        if current_position and order.symbol in market_prices:
                            current_value = current_position["shares"] * market_prices[order.symbol]
                        
                        buy_amount = target_value - current_value
                        
                        if buy_amount > 0:
                            engine.buy(order.symbol, amount=buy_amount)
                            logger.info(
                                f"[{self.name}] 执行买入订单: {order.symbol}, "
                                f"目标权重 {order.target_weight:.2%}, "
                                f"买入金额 {buy_amount:.2f}, "
                                f"置信度 {order.confidence:.2f}"
                            )
                
                elif order.side == OrderSide.SELL:
                    if order.qty is not None and order.qty > 0:
                        engine.sell(order.symbol, shares=order.qty)
                        logger.info(
                            f"[{self.name}] 执行卖出订单: {order.symbol}, "
                            f"股数 {order.qty}"
                        )
            
            except Exception as e:
                logger.error(f"执行订单时出错: {order}, 错误: {e}", exc_info=True)
    
    def on_date(self, engine: BacktestEngine, date: str):
        """
        每个交易日的回调函数
        
        实现带 LLM 不确定性评估的定投策略：
        1. 判断是否应该定投
        2. 使用 LLM 评估不确定性
        3. 生成订单意图（包含 confidence）
        4. 应用风险控制
        5. 执行订单
        
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
            
            # 生成订单意图列表
            order_intents = []
            
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
                    
                    # 使用 LLM 评估不确定性
                    logger.info(f"[{self.name}] 开始 LLM 不确定性评估: {stock_code} on {date}")
                    llm_result = self._evaluate_uncertainty(stock_code, engine, date)
                    
                    sentiment = llm_result.get("sentiment", 0.0)
                    confidence = llm_result.get("confidence", 0.5)
                    reasoning = llm_result.get("reasoning", "")
                    
                    # 如果 sentiment 为负，不执行买入（可选：也可以降低仓位）
                    if sentiment < 0:
                        logger.info(
                            f"[{self.name}] LLM 评估 sentiment 为负 ({sentiment:.2f})，跳过买入: {stock_code}"
                        )
                        continue
                    
                    # 计算买入股数（基于固定金额）
                    shares_to_buy = int(investment_per_stock / price)
                    
                    if shares_to_buy > 0:
                        # 生成买入订单意图（包含 confidence）
                        order_intent = OrderIntent(
                            symbol=stock_code,
                            side=OrderSide.BUY,
                            timestamp=date,
                            qty=shares_to_buy,
                            price_type=PriceType.MKT,
                            agent_name=self.name,
                            confidence=confidence,  # 使用 LLM 评估的置信度
                            reason_tags=["llm_evaluated"],
                            metadata={
                                "dca_amount": investment_per_stock,
                                "llm_sentiment": sentiment,
                                "llm_reasoning": reasoning
                            }
                        )
                        order_intents.append(order_intent)
                    else:
                        logger.warning(
                            f"[{date}] 定投金额 {investment_per_stock:.2f} 元无法买入至少1股 "
                            f"(价格: {price:.2f})，跳过定投"
                        )
                    
                except Exception as e:
                    logger.error(f"[{date}] 生成 {stock_code} 订单意图时出错: {e}", exc_info=True)
            
            if not order_intents:
                return
            
            # 应用风险控制
            approved_orders = self.apply_risk_control(order_intents, engine)
            
            # 过滤掉被削减为 0 的订单
            valid_orders = []
            for order in approved_orders:
                if order.qty is not None and order.qty > 0:
                    valid_orders.append(order)
                elif order.target_weight is not None and order.target_weight > 0:
                    valid_orders.append(order)
            
            if not valid_orders:
                logger.info(
                    f"[{date}] 所有定投订单都被风险控制拒绝，跳过本次定投"
                )
                return
            
            # 执行订单
            self.execute_orders(valid_orders, engine)
            
            # 更新定投次数
            self.investment_count += len(order_intents)
            
        except Exception as e:
            logger.error(f"[{date}] 执行定投策略时出错: {e}", exc_info=True)

