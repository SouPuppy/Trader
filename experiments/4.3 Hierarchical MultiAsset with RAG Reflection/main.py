"""
实验 4.3：层级式多资产交易系统（带 RAG 反思层）

Layer A: Signal Layer（信号层）
Layer B: Allocation & RiskControl Layer（分配与风控层）
Layer C: RAG 反思层（参数θ每周调整，使用 RAG 系统）

在 4.2 的基础上，使用 RAG 系统进行反思：
1. 评估过去一周的表现（周收益率、交易情况、市场状态）
2. 使用 RAG 系统检索历史数据、新闻、趋势等多维度信息
3. 基于 RAG 检索的证据生成参数调整建议
4. 解析 RAG 答案并应用参数调整

优化内容（继承自 4.1 和 4.2）：
1. 实现新闻信号功能：使用最基础的features（news_count），朴素算法，不做复杂分析
2. 改进趋势信号：使用多时间框架（1日、5日、20日）加权组合，捕捉趋势持续性
3. 优化基本面信号：综合PE、PB、PS多个估值指标，取平均值
4. 改进波动率信号：使用相对波动率（20日/60日），而非绝对波动率
5. 调整信号权重：提高趋势权重至0.5，降低波动率和新闻权重至0.15
6. 优化策略参数：
   - gross_exposure: 1.0 -> 0.7（降低总仓位，更保守）
   - max_w: 0.20 -> 0.15（降低单票上限，分散风险）
   - turnover_cap: 0.30 -> 0.20（进一步降低换手率，减少交易成本）
   - enter_th: 0.0 -> 0.02（降低进场阈值，让更多股票有机会）
   - exit_th: -0.1 -> -0.10（放宽出场阈值，避免过早止损）
7. 优化交易阈值：从固定100元改为账户权益的0.5%，减少不必要交易
"""
import sys
import json
import re
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict

# 添加项目根目录到 Python 路径
project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from trader.agent.theta import Theta
from trader.agent.signal_layer import SignalLayer
from trader.agent.constrained_allocator import ConstrainedAllocator
from trader.backtest.account import Account
from trader.backtest.market import Market
from trader.backtest.reflection_engine import ReflectionEngine
from trader.logger import get_logger, log_separator, log_section

logger = get_logger(__name__)


def is_weekend_reflection_day(date_str: str) -> bool:
    """
    判断是否是周末反思日（周五）
    
    Args:
        date_str: 日期字符串，格式为 "YYYY-MM-DD"
        
    Returns:
        bool: 是否是周五
    """
    try:
        date_obj = datetime.strptime(date_str, "%Y-%m-%d")
        # 0 = Monday, 4 = Friday
        return date_obj.weekday() == 4
    except Exception as e:
        logger.warning(f"无法解析日期 {date_str}: {e}")
        return False


def build_reflection_question(
    current_theta: Theta,
    weekly_return: float,
    account_equity: float,
    initial_cash: float,
    stock_codes: list,
    reflection_id: int
) -> str:
    """
    构建 RAG 反思问题
    
    Args:
        current_theta: 当前参数θ
        weekly_return: 周收益率（小数）
        account_equity: 当前账户权益
        initial_cash: 初始资金
        stock_codes: 股票代码列表
        reflection_id: 反思轮次ID
        
    Returns:
        str: RAG 问题文本
    """
    total_return = (account_equity - initial_cash) / initial_cash if initial_cash > 0 else 0.0
    
    question = f"""作为专业的量化交易策略分析师，请基于以下信息评估多资产组合交易策略的表现，并提供参数调整建议。

【当前策略参数（θ）】
- 总仓位上限 (gross_exposure): {current_theta.gross_exposure:.2f}
- 单票上限 (max_w): {current_theta.max_w:.2f}
- 换手率上限 (turnover_cap): {current_theta.turnover_cap:.2f}
- 风险模式 (risk_mode): {current_theta.risk_mode}
- 进场阈值 (enter_th): {current_theta.enter_th:.3f}
- 出场阈值 (exit_th): {current_theta.exit_th:.3f}

【交易表现】
- 本周收益率: {weekly_return*100:.2f}%
- 累计收益率: {total_return*100:.2f}%
- 当前账户权益: {account_equity:,.2f} 元
- 初始资金: {initial_cash:,.2f} 元
- 交易股票池（多资产组合）: {', '.join(stock_codes)}

【分析要求】
请基于 RAG 系统检索到的历史数据、新闻、趋势等多维度信息，分析：
1. 本周表现评估：本周收益率 {weekly_return*100:.2f}% 是否良好？与市场整体表现相比如何？
   **重要**：即使检索到的数据只覆盖部分股票，也要基于周收益率评估整体策略表现
   **重要**：周收益率是评估策略表现的核心指标，必须优先考虑
2. 市场环境分析：当前市场环境如何？组合中的股票（{', '.join(stock_codes)}）有哪些共同的风险或机会？
   **重要**：虽然检索数据可能只针对单只股票，但要分析整个多资产组合的共同特征
   **重要**：如果检索数据不足，可以基于周收益率推断市场环境（正收益可能表示市场向好，负收益可能表示市场承压）
3. 参数调整建议：基于市场环境和策略表现，策略参数是否需要调整？如何调整才能提升收益或降低风险？
   **重要**：即使证据不足，也要基于周收益率给出调整建议，而非完全拒绝调整
   **重要**：如果周收益率 > 1.5%，建议适度提高仓位或降低进场阈值；如果周收益率 < -1.5%，建议适度降低仓位或提高进场阈值
   **重要**：不要因为证据不足就拒绝调整，应该基于策略表现（周收益率）给出保守但合理的建议

【输出格式】
请以 JSON 格式返回参数调整建议，格式如下：
{{
    "analysis": "简要分析本周表现和市场环境（基于检索到的证据）",
    "adjustments": {{
        "gross_exposure": 调整值或null（不调整）,
        "max_w": 调整值或null,
        "turnover_cap": 调整值或null,
        "risk_mode": "risk_on" | "neutral" | "risk_off" 或null,
        "enter_th": 调整值或null,
        "exit_th": 调整值或null
    }},
    "reasoning": "调整理由的详细说明（应引用检索到的证据）"
}}

【重要约束】
- 参数调整必须保守且渐进，避免大幅波动
- **重要**：即使证据不足，也要基于周收益率（{weekly_return*100:.2f}%）给出参数调整建议
- **重要**：如果证据不足，应该基于策略表现（周收益率）进行适度调整，而非完全拒绝调整
- **重要**：分析整个多资产组合的表现，而非单只股票
- 参数范围限制：
  * gross_exposure: [0.4, 0.85]
  * max_w: [0.08, 0.18]
  * turnover_cap: [0.10, 0.35]
  * enter_th: [0.01, 0.05]
  * exit_th: [-0.15, -0.08]
- 如果某个参数不需要调整，必须设置为 null（不要使用当前值）
- **调整原则**：
  * 如果周收益率 > 1.5%，可以适度提高仓位（gross_exposure, max_w）或降低进场阈值（enter_th）
  * 如果周收益率 < -1.5%，可以适度降低仓位或提高进场阈值
  * 如果周收益率在 ±1.5% 范围内，可以保持参数不变或微调
"""
    return question


def parse_rag_adjustments(rag_answer: str, current_theta: Theta) -> Optional[Dict]:
    """
    解析 RAG 返回的答案，提取参数调整建议
    
    Args:
        rag_answer: RAG 系统返回的答案文本
        current_theta: 当前参数θ
        
    Returns:
        Dict: 包含调整后参数的字典，如果解析失败返回 None
    """
    try:
        # 尝试提取 JSON 部分
        json_match = re.search(r'\{[^{}]*"adjustments"[^{}]*\{[^{}]*\}[^{}]*\}', rag_answer, re.DOTALL)
        if json_match:
            json_str = json_match.group(0)
        else:
            # 如果没有找到 JSON，尝试提取整个 JSON 对象
            json_match = re.search(r'\{.*\}', rag_answer, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
            else:
                logger.warning(f"无法从 RAG 答案中提取 JSON: {rag_answer[:200]}...")
                return None
        
        # 解析 JSON
        data = json.loads(json_str)
        adjustments = data.get("adjustments", {})
        
        # 构建新的参数
        new_theta_dict = {
            "gross_exposure": adjustments.get("gross_exposure", current_theta.gross_exposure),
            "max_w": adjustments.get("max_w", current_theta.max_w),
            "turnover_cap": adjustments.get("turnover_cap", current_theta.turnover_cap),
            "risk_mode": adjustments.get("risk_mode", current_theta.risk_mode),
            "enter_th": adjustments.get("enter_th", current_theta.enter_th),
            "exit_th": adjustments.get("exit_th", current_theta.exit_th),
        }
        
        # 处理 null 值（保持原值）
        for key in ["gross_exposure", "max_w", "turnover_cap", "enter_th", "exit_th"]:
            if adjustments.get(key) is None:
                new_theta_dict[key] = getattr(current_theta, key)
        
        if adjustments.get("risk_mode") is None:
            new_theta_dict["risk_mode"] = current_theta.risk_mode
        
        # 参数范围校验
        new_theta_dict["gross_exposure"] = max(0.4, min(0.85, new_theta_dict["gross_exposure"]))
        new_theta_dict["max_w"] = max(0.08, min(0.18, new_theta_dict["max_w"]))
        new_theta_dict["turnover_cap"] = max(0.10, min(0.35, new_theta_dict["turnover_cap"]))
        new_theta_dict["enter_th"] = max(0.01, min(0.05, new_theta_dict["enter_th"]))
        new_theta_dict["exit_th"] = max(-0.15, min(-0.08, new_theta_dict["exit_th"]))
        
        # 添加分析信息
        analysis = data.get("analysis", "")
        reasoning = data.get("reasoning", "")
        
        return {
            "theta": new_theta_dict,
            "analysis": analysis,
            "reasoning": reasoning
        }
        
    except json.JSONDecodeError as e:
        logger.error(f"解析 RAG JSON 失败: {e}")
        logger.debug(f"原始答案: {rag_answer[:500]}")
        return None
    except Exception as e:
        logger.error(f"解析 RAG 答案时出错: {e}", exc_info=True)
        return None


def rag_reflection(
    current_theta: Theta,
    weekly_return: float,
    account_equity: float,
    initial_cash: float,
    stock_codes: list,
    reflection_id: int,
    decision_time: str,
    llm_model: str = "deepseek-chat"
) -> Theta:
    """
    RAG 反思：使用 RAG 系统生成参数调整建议
    
    Args:
        current_theta: 当前参数θ
        weekly_return: 周收益率（小数）
        account_equity: 当前账户权益
        initial_cash: 初始资金
        stock_codes: 股票代码列表
        reflection_id: 反思轮次ID
        decision_time: 决策时间（ISO8601 格式）
        llm_model: LLM 模型名称
        
    Returns:
        Theta: 调整后的参数θ
    """
    try:
        from trader.rag.answer import rag_answer
    except ImportError:
        logger.error("无法导入 RAG 模块，回退到 naive reflection")
        # 回退到简单的调整逻辑
        return naive_reflection_fallback(current_theta, weekly_return, reflection_id)
    
    # 构建反思问题
    question = build_reflection_question(
        current_theta=current_theta,
        weekly_return=weekly_return,
        account_equity=account_equity,
        initial_cash=initial_cash,
        stock_codes=stock_codes,
        reflection_id=reflection_id
    )
    
    logger.info(f"[RAG Reflection] 开始 RAG 反思（第 {reflection_id} 次）")
    logger.info(f"[RAG Reflection] 周收益率: {weekly_return*100:.2f}%")
    logger.info(f"[RAG Reflection] 交易股票池: {', '.join(stock_codes)}")
    
    # 改进：尝试为多只股票检索数据，而非只用第一只
    # 优先选择有交易历史的股票
    primary_stock = stock_codes[0] if stock_codes else None
    
    # 尝试选择有更多数据的股票（如果有交易历史更好）
    # 这里简化处理，使用第一只股票，但在 prompt 中强调多资产分析
    if not primary_stock:
        logger.warning("[RAG Reflection] 没有股票代码，回退到 naive reflection")
        return naive_reflection_fallback(current_theta, weekly_return, reflection_id)
    
    try:
        logger.info(f"[RAG Reflection] 调用 RAG 系统，使用股票代码: {primary_stock}")
        logger.info(f"[RAG Reflection] 注意：RAG 将基于 {primary_stock} 检索数据，但分析应涵盖整个组合: {', '.join(stock_codes)}")
        verified_answer = rag_answer(
            question=question,
            stock_code=primary_stock,
            decision_time=decision_time,
            frequency="1d"
        )
        
        if not verified_answer:
            logger.warning("[RAG Reflection] RAG 系统返回空结果，回退到 naive reflection")
            return naive_reflection_fallback(current_theta, weekly_return, reflection_id)
        
        logger.info(f"[RAG Reflection] RAG 系统调用成功:")
        logger.info(f"  - 验证状态: {'通过' if verified_answer.passed else '未通过'}")
        logger.info(f"  - 模式: {verified_answer.mode}")
        logger.info(f"  - 使用的文档数: {len(verified_answer.used_doc_ids)}")
        if verified_answer.violations:
            logger.warning(f"  - 验证违规: {verified_answer.violations}")
        
        # 检查是否是 degraded mode 或证据不足
        is_degraded = verified_answer.mode == "degraded"
        has_insufficient_evidence = any(
            phrase in verified_answer.answer.lower() 
            for phrase in ["insufficient evidence", "cannot answer", "not enough evidence", "no parameter adjustments"]
        )
        
        # 解析 RAG 答案
        adjustments = parse_rag_adjustments(verified_answer.answer, current_theta)
        
        # 检查 RAG 是否拒绝调整（所有参数都是 null 或保持原值）
        rag_refused_adjustment = False
        if adjustments:
            rag_theta = adjustments["theta"]
            # 检查是否所有参数都保持原值
            if (rag_theta["gross_exposure"] == current_theta.gross_exposure and
                rag_theta["max_w"] == current_theta.max_w and
                rag_theta["turnover_cap"] == current_theta.turnover_cap and
                rag_theta["risk_mode"] == current_theta.risk_mode and
                rag_theta["enter_th"] == current_theta.enter_th and
                rag_theta["exit_th"] == current_theta.exit_th):
                rag_refused_adjustment = True
        
        # 如果 degraded mode、证据不足或拒绝调整，使用改进的混合策略
        if is_degraded or has_insufficient_evidence or not adjustments or rag_refused_adjustment:
            logger.warning(f"[RAG Reflection] RAG 系统状态: degraded={is_degraded}, insufficient_evidence={has_insufficient_evidence}, refused={rag_refused_adjustment}")
            logger.info("[RAG Reflection] 使用改进的混合策略：优先使用 naive reflection，RAG 仅作为参考")
            
            # 优先使用 naive reflection（基于周收益率，更可靠）
            naive_theta = naive_reflection_fallback(current_theta, weekly_return, reflection_id)
            
            # 如果 RAG 有部分建议，尝试融合（改进：采用更合理的值，而非总是更保守）
            if adjustments and not rag_refused_adjustment:
                rag_theta = adjustments["theta"]
                
                # 检查方向一致性
                rag_gross_change = rag_theta["gross_exposure"] - current_theta.gross_exposure
                naive_gross_change = naive_theta.gross_exposure - current_theta.gross_exposure
                
                # 改进：如果方向一致，采用两者的平均值（更平衡），而非总是更保守的值
                if rag_gross_change * naive_gross_change > 0:  # 同方向
                    # 采用平均值，但偏向 naive（因为 naive 基于周收益率，更可靠）
                    weight_naive = 0.7  # naive 权重更高
                    weight_rag = 0.3
                    naive_theta.gross_exposure = (
                        weight_naive * naive_theta.gross_exposure + 
                        weight_rag * rag_theta["gross_exposure"]
                    )
                    naive_theta.max_w = (
                        weight_naive * naive_theta.max_w + 
                        weight_rag * rag_theta["max_w"]
                    )
                    # 改进：提高仓位下限，避免过度保守
                    # 确保在合理范围内，但下限更高
                    naive_theta.gross_exposure = max(0.60, min(0.85, naive_theta.gross_exposure))  # 提高下限
                    naive_theta.max_w = max(0.12, min(0.18, naive_theta.max_w))  # 提高下限
                    logger.info("[RAG Reflection] 融合 RAG 和 naive 建议（采用加权平均，偏向 naive，设置更高下限）")
                else:
                    # 方向不一致，采用 naive（因为 naive 基于周收益率，更可靠）
                    logger.info("[RAG Reflection] RAG 和 naive 方向不一致，采用 naive 建议")
            
            logger.info(f"[RAG Reflection] 混合策略最终参数: {current_theta} -> {naive_theta}")
            return naive_theta
        
        # RAG 正常模式且有有效调整
        new_theta = Theta(
            gross_exposure=adjustments["theta"]["gross_exposure"],
            max_w=adjustments["theta"]["max_w"],
            turnover_cap=adjustments["theta"]["turnover_cap"],
            risk_mode=adjustments["theta"]["risk_mode"],
            enter_th=adjustments["theta"]["enter_th"],
            exit_th=adjustments["theta"]["exit_th"],
            reflection_id=reflection_id
        )
        
        logger.info(f"[RAG Reflection] RAG 分析: {adjustments.get('analysis', 'N/A')[:200]}...")
        logger.info(f"[RAG Reflection] 调整理由: {adjustments.get('reasoning', 'N/A')[:200]}...")
        logger.info(f"[RAG Reflection] 参数调整: {current_theta} -> {new_theta}")
        
        return new_theta
        
    except Exception as e:
        logger.error(f"[RAG Reflection] 调用 RAG 系统时出错: {e}", exc_info=True)
        logger.warning("[RAG Reflection] 回退到 naive reflection")
        return naive_reflection_fallback(current_theta, weekly_return, reflection_id)


def naive_reflection_fallback(current_theta: Theta, weekly_return: float, reflection_id: int) -> Theta:
    """
    改进的反思回退方案：更积极的调整策略
    
    Args:
        current_theta: 当前参数θ
        weekly_return: 周收益率（小数）
        reflection_id: 反思轮次ID
        
    Returns:
        Theta: 调整后的参数θ
    """
    new_theta = current_theta.copy()
    new_theta.reflection_id = reflection_id
    
    # 改进：降低阈值，更积极的调整策略
    adjustment_factor = 0.02  # 2% 的调整幅度
    positive_threshold = 0.01  # 1.0%（降低阈值，更积极）
    negative_threshold = -0.01  # -1.0%
    
    if weekly_return > positive_threshold:  # 周收益率 > 1.0%，表现良好
        logger.info(f"[Naive Reflection Fallback] 周收益率 {weekly_return*100:.2f}% > {positive_threshold*100:.2f}%，表现良好，适度激进调整")
        # 改进：更积极地提高仓位，充分利用盈利机会
        # 使用更大的调整幅度，确保仓位能充分利用盈利
        aggressive_factor = adjustment_factor * 1.5  # 1.5倍调整幅度
        new_theta.gross_exposure = min(0.85, current_theta.gross_exposure * (1 + aggressive_factor))
        new_theta.max_w = min(0.20, current_theta.max_w * (1 + aggressive_factor))
        new_theta.turnover_cap = min(0.30, current_theta.turnover_cap * (1 + adjustment_factor * 0.6))
        if current_theta.risk_mode == "risk_off":
            new_theta.risk_mode = "neutral"
        new_theta.enter_th = max(0.01, current_theta.enter_th * (1 - adjustment_factor * 0.8))
        # 改进：不要过度放宽 exit_th，保持合理的止损
        new_theta.exit_th = min(-0.08, current_theta.exit_th * (1 - adjustment_factor * 0.4))  # 降低调整幅度
    elif weekly_return < negative_threshold:  # 周收益率 < -1.0%，表现不佳
        logger.info(f"[Naive Reflection Fallback] 周收益率 {weekly_return*100:.2f}% < {negative_threshold*100:.2f}%，表现不佳，更保守调整")
        # 改进：设置更高的仓位下限，避免过度保守
        # 提高下限：从 0.50 提高到 0.60，从 0.10 提高到 0.12
        min_gross_exposure = 0.60  # 提高下限
        min_max_w = 0.12  # 提高下限
        new_theta.gross_exposure = max(min_gross_exposure, current_theta.gross_exposure * (1 - adjustment_factor))
        new_theta.max_w = max(min_max_w, current_theta.max_w * (1 - adjustment_factor))
        new_theta.turnover_cap = max(0.15, current_theta.turnover_cap * (1 - adjustment_factor))
        if current_theta.risk_mode == "risk_on":
            new_theta.risk_mode = "neutral"
        elif current_theta.risk_mode == "neutral":
            new_theta.risk_mode = "risk_off"
        new_theta.enter_th = min(0.04, current_theta.enter_th * (1 + adjustment_factor * 0.8))
        # 改进：不要过度收紧 exit_th，保持合理的止损范围
        new_theta.exit_th = max(-0.12, current_theta.exit_th * (1 - adjustment_factor * 0.4))  # 降低调整幅度
    else:
        # 在 ±1.0% 范围内，微调而非完全不变
        # 根据收益率的正负进行微调
        if weekly_return > 0:
            # 小幅正收益，微调提高仓位
            logger.info(f"[Naive Reflection Fallback] 周收益率 {weekly_return*100:.2f}% 在正常波动范围（±{positive_threshold*100:.2f}%），小幅正收益，微调提高仓位")
            new_theta.gross_exposure = min(0.85, current_theta.gross_exposure * (1 + adjustment_factor * 0.5))
            new_theta.max_w = min(0.20, current_theta.max_w * (1 + adjustment_factor * 0.5))
        elif weekly_return < 0:
            # 小幅负收益，微调降低仓位（但设置下限，避免过度保守）
            logger.info(f"[Naive Reflection Fallback] 周收益率 {weekly_return*100:.2f}% 在正常波动范围（±{positive_threshold*100:.2f}%），小幅负收益，微调降低仓位")
            min_gross_exposure = 0.60  # 提高下限
            min_max_w = 0.12  # 提高下限
            new_theta.gross_exposure = max(min_gross_exposure, current_theta.gross_exposure * (1 - adjustment_factor * 0.5))
            new_theta.max_w = max(min_max_w, current_theta.max_w * (1 - adjustment_factor * 0.5))
        else:
            # 收益率为 0，保持参数不变
            logger.info(f"[Naive Reflection Fallback] 周收益率 {weekly_return*100:.2f}% 接近 0，保持参数不变")
    
    logger.info(f"[Naive Reflection Fallback] 参数调整: {current_theta} -> {new_theta}")
    return new_theta


def hierarchical_multiasset_strategy_with_rag_reflection(
    stock_codes: list = None,
    initial_cash: float = 1000000.0,
    initial_theta: Theta = None,
    start_date: str = None,
    end_date: str = None,
    train_test_split_ratio: float = 0.7,
    llm_model: str = "deepseek-chat"
):
    """
    层级式多资产交易策略（带 RAG 反思层）
    
    Args:
        stock_codes: 股票代码列表
        initial_cash: 初始资金
        initial_theta: 初始参数θ（如果为None则使用默认值）
        start_date: 开始日期
        end_date: 结束日期
        train_test_split_ratio: 训练/测试分割比例
        llm_model: LLM 模型名称（RAG 系统内部使用）
    """
    if stock_codes is None:
        stock_codes = [
            "AAPL.O", "MSFT.O", "GOOGL.O", "AMZN.O", "NVDA.O",
            "TSLA.O", "META.O", "ASML.O", "MRNA.O", "NFLX.O",
            "AMD.O", "INTC.O", "ADBE.O", "CRM.N", "ORCL.N",
            "CSCO.O", "JPM.N", "V.N", "MA.N", "WMT.N"
        ]
    
    if initial_theta is None:
        initial_theta = Theta(
            gross_exposure=0.85,  # 提高总仓位到85%，充分利用资金
            max_w=0.20,  # 单票上限20%，允许集中配置优质股票
            turnover_cap=0.25,  # 适度提高换手率上限，允许灵活调整
            risk_mode="neutral",
            enter_th=0.02,  # 降低进场阈值，让更多股票有机会
            exit_th=-0.10  # 放宽出场阈值，避免过早止损
        )
    
    for line in log_section("层级式多资产交易系统（带 RAG 反思层）"):
        logger.info(line)
    logger.info(f"股票代码: {', '.join(stock_codes)}")
    logger.info(f"初始资金: {initial_cash:,.2f} 元")
    logger.info(f"初始参数θ: {initial_theta}")
    logger.info(f"RAG 系统 LLM 模型: {llm_model}")
    logger.info("")
    logger.info("RAG 反思说明:")
    logger.info("  - 每周五使用 RAG 系统检索历史数据、新闻、趋势等多维度信息")
    logger.info("  - 基于检索到的证据分析交易表现和市场环境")
    logger.info("  - 生成参数调整建议并应用")
    logger.info("")
    
    # 初始化市场、账户和回测引擎
    market = Market(price_adjustment=0.01)
    account = Account(initial_cash=initial_cash)
    
    # 设置输出目录
    experiment_dir = Path(__file__).parent.name
    from trader.config import PROJECT_ROOT
    output_dir = PROJECT_ROOT / 'output' / 'backtest' / experiment_dir
    
    # 创建ReflectionEngine（支持参数追踪）
    engine = ReflectionEngine(
        account=account,
        market=market,
        initial_theta=initial_theta,
        report_title=None,  # 不自动生成报告
        report_output_dir=output_dir,
        train_test_split_ratio=train_test_split_ratio,
        only_test_period=False  # 从第一天开始交易，不限制训练期
    )
    
    # 获取可用日期
    available_dates = market.get_available_dates(stock_codes[0])
    if not available_dates:
        logger.error(f"未找到股票数据")
        return
    
    logger.info(f"数据范围: {available_dates[0]} 至 {available_dates[-1]}")
    
    # 创建Layer A: Signal Layer（优化权重）
    signal_layer = SignalLayer(
        name="SignalLayer",
        w_T=0.5,
        w_V=0.15,
        w_F=0.20,
        w_N=0.15,
        theta=initial_theta
    )
    
    # 创建Layer B: Constrained Allocator
    allocator = ConstrainedAllocator(
        signal_layer=signal_layer,
        theta=initial_theta,
        name="ConstrainedAllocator"
    )
    
    # 跟踪每周权益（用于计算周收益率）
    weekly_equity_history: Dict[str, float] = {}
    last_reflection_date: Optional[str] = None
    reflection_count = 0
    
    # 注册交易日回调
    def on_trading_day(eng: ReflectionEngine, date: str):
        """每个交易日的回调"""
        nonlocal last_reflection_date, reflection_count
        
        # 记录每日参数（即使没有变化）
        eng.record_daily_theta(date)
        
        # 获取账户权益
        market_prices = eng.get_market_prices(stock_codes)
        account_equity = account.equity(market_prices)
        
        # 记录每日权益
        weekly_equity_history[date] = account_equity
        
        # 检查是否是周五（周末反思日）
        if is_weekend_reflection_day(date):
            # 计算周收益率：优先使用账户权益，如果没有交易则使用市场表现
            weekly_return = 0.0
            use_market_return = False
            
            # 找到上周五的权益（或本周第一个交易日）
            if last_reflection_date and last_reflection_date in weekly_equity_history:
                last_equity = weekly_equity_history[last_reflection_date]
                if last_equity and last_equity > 0:
                    weekly_return = (account_equity - last_equity) / last_equity
                    logger.info(f"[Reflection] 计算周收益率（账户权益）: {last_reflection_date} -> {date}")
                    logger.info(f"[Reflection] 上周五权益: {last_equity:,.2f}, 本周五权益: {account_equity:,.2f}")
                    logger.info(f"[Reflection] 周收益率: {weekly_return*100:.2f}%")
                else:
                    use_market_return = True
            else:
                # 第一次反思，检查是否有交易
                if account.initial_cash > 0 and account_equity != account.initial_cash:
                    weekly_return = (account_equity - account.initial_cash) / account.initial_cash
                    logger.info(f"[Reflection] 第一次反思，使用账户权益作为基准")
                    logger.info(f"[Reflection] 初始资金: {account.initial_cash:,.2f}, 当前权益: {account_equity:,.2f}")
                    logger.info(f"[Reflection] 周收益率: {weekly_return*100:.2f}%")
                else:
                    use_market_return = True
            
            # 如果没有交易或权益没有变化，使用市场表现
            if use_market_return:
                market_return = 0.0
                valid_stocks = 0
                
                if last_reflection_date:
                    for stock_code in stock_codes:
                        try:
                            last_price = eng.get_price(stock_code, last_reflection_date)
                            current_price = eng.get_price(stock_code, date)
                            if last_price and current_price and last_price > 0:
                                stock_return = (current_price - last_price) / last_price
                                market_return += stock_return
                                valid_stocks += 1
                        except Exception as e:
                            logger.debug(f"无法获取 {stock_code} 的价格: {e}")
                    
                    if valid_stocks > 0:
                        market_return = market_return / valid_stocks
                        weekly_return = market_return
                        logger.info(f"[Reflection] 计算周收益率（市场表现）: {last_reflection_date} -> {date}")
                        logger.info(f"[Reflection] 组合平均周收益率: {weekly_return*100:.2f}% (基于 {valid_stocks} 只股票)")
                    else:
                        logger.warning(f"[Reflection] 无法计算市场收益率，跳过本次反思")
                        return
                else:
                    # 第一次反思，使用第一周的市场表现
                    if available_dates and len(available_dates) > 5:
                        first_date = available_dates[0]
                        try:
                            for stock_code in stock_codes:
                                first_price = eng.get_price(stock_code, first_date)
                                current_price = eng.get_price(stock_code, date)
                                if first_price and current_price and first_price > 0:
                                    stock_return = (current_price - first_price) / first_price
                                    market_return += stock_return
                                    valid_stocks += 1
                            
                            if valid_stocks > 0:
                                market_return = market_return / valid_stocks
                                weekly_return = market_return
                                logger.info(f"[Reflection] 第一次反思，使用市场表现作为基准")
                                logger.info(f"[Reflection] 组合平均周收益率: {weekly_return*100:.2f}% (基于 {valid_stocks} 只股票)")
                            else:
                                logger.warning(f"[Reflection] 无法计算市场收益率，跳过本次反思")
                                return
                        except Exception as e:
                            logger.warning(f"[Reflection] 计算市场收益率时出错: {e}")
                            return
                    else:
                        logger.warning(f"[Reflection] 数据不足，跳过本次反思")
                        return
            
            # 执行 RAG 反思
            reflection_count += 1
            current_theta = eng.get_current_theta()
            
            # 构建决策时间（ISO8601 格式）
            decision_time = f"{date}T00:00:00"
            
            new_theta = rag_reflection(
                current_theta=current_theta,
                weekly_return=weekly_return,
                account_equity=account_equity,
                initial_cash=account.initial_cash,
                stock_codes=stock_codes,
                reflection_id=reflection_count,
                decision_time=decision_time,
                llm_model=llm_model
            )
            
            # 更新参数
            eng.update_theta(new_theta, date)
            
            # 更新 allocator 和 signal_layer 的参数
            allocator.update_theta(new_theta)
            signal_layer.update_theta(new_theta)
            
            # 记录本次反思日期
            last_reflection_date = date
        
        # 获取所有股票的目标权重（使用更新后的参数）
        target_weights = allocator.get_weights(stock_codes, eng)
        
        # 执行交易：调整到目标权重
        for stock_code in stock_codes:
            target_weight = target_weights.get(stock_code, 0.0)
            target_value = account_equity * target_weight
            
            # 获取当前持仓
            position = account.get_position(stock_code)
            current_value = 0.0
            current_price = eng.get_current_price(stock_code)
            if position and current_price:
                current_value = position['shares'] * current_price
            
            # 计算需要调整的金额
            diff_value = target_value - current_value
            
            # 如果变化很小（小于账户权益的0.5%），不需要交易
            min_trade_threshold = account_equity * 0.005
            if abs(diff_value) < min_trade_threshold:
                continue
            
            # 执行交易
            if diff_value > 0:
                # 买入
                eng.buy(stock_code, amount=diff_value)
            else:
                # 卖出
                shares_to_sell = int(abs(diff_value) / current_price) if current_price else 0
                if shares_to_sell > 0 and position:
                    shares_to_sell = min(shares_to_sell, position['shares'])
                    if shares_to_sell > 0:
                        eng.sell(stock_code, shares=shares_to_sell)
        
        # Balance 要求：确保大量资金都在股市里（现金比例不超过10%）
        # 如果现金比例过高，按权重比例买入股票
        # 重新获取账户权益（交易后可能已变化）
        market_prices_after_trade = eng.get_market_prices(stock_codes)
        account_equity_after_trade = account.equity(market_prices_after_trade)
        cash_ratio = account.cash / account_equity_after_trade if account_equity_after_trade > 0 else 1.0
        max_cash_ratio = 0.10  # 最多保留10%现金
        
        if cash_ratio > max_cash_ratio and account.cash > 0:
            # 计算需要投入的资金
            excess_cash = account.cash - (account_equity_after_trade * max_cash_ratio)
            min_trade_threshold_balance = account_equity_after_trade * 0.005
            if excess_cash > min_trade_threshold_balance:
                logger.info(f"[Balance] 现金比例 {cash_ratio*100:.2f}% 过高，需要投入 {excess_cash:,.2f} 元到股市")
                
                # 按当前权重比例分配多余现金
                total_weight = sum(target_weights.values())
                if total_weight > 0:
                    for stock_code in stock_codes:
                        weight = target_weights.get(stock_code, 0.0)
                        if weight > 0:
                            # 按权重分配资金
                            invest_amount = excess_cash * (weight / total_weight)
                            if invest_amount > min_trade_threshold_balance:
                                eng.buy(stock_code, amount=invest_amount)
                                logger.debug(f"[Balance] 买入 {stock_code}: {invest_amount:,.2f} 元")
    
    engine.on_date(on_trading_day)
    
    # 运行回测
    logger.info("")
    logger.info("开始回测...")
    engine.run(stock_codes[0], start_date=start_date, end_date=end_date)
    
    # 计算最终结果
    final_date = engine.current_date if engine.current_date else None
    if not final_date:
        final_date = end_date if end_date else available_dates[-1]
    
    market_prices = {}
    for stock_code in stock_codes:
        price = market.get_price(stock_code, final_date)
        if price is None:
            price = market.get_price(stock_code)
        if price:
            market_prices[stock_code] = price
    
    if not market_prices:
        logger.error("无法获取最终价格")
        return
    
    equity = account.equity(market_prices)
    profit = account.get_total_profit(market_prices)
    return_pct = account.get_total_return(market_prices)
    
    # 输出结果
    logger.info("")
    for line in log_section("回测结果"):
        logger.info(line)
    logger.info(f"初始资金: {initial_cash:,.2f} 元")
    logger.info(f"最终权益: {equity:,.2f} 元")
    logger.info(f"总盈亏: {profit:+,.2f} 元")
    logger.info(f"总收益率: {return_pct:+.2f}%")
    logger.info(f"交易次数: {len(account.trades)}")
    logger.info(f"参数变化次数: {len(engine.theta_history)}")
    logger.info(f"反思次数: {reflection_count}")
    logger.info(log_separator())
    
    # 生成参数化报告
    logger.info("")
    logger.info("生成参数化报告...")
    actual_start_date = start_date or available_dates[0]
    actual_end_date = end_date or available_dates[-1]
    
    report_file = engine.generate_parametrized_report(
        stock_codes=stock_codes,
        start_date=actual_start_date,
        end_date=actual_end_date,
        strategy_name="层级式多资产交易系统（带 RAG 反思层）"
    )
    
    logger.info(f"报告已保存: {report_file}")


if __name__ == "__main__":
    # 执行层级式多资产交易策略（带 RAG 反思层）
    hierarchical_multiasset_strategy_with_rag_reflection(
        stock_codes=[
            "AAPL.O", "MSFT.O", "GOOGL.O", "AMZN.O", "NVDA.O",
            "TSLA.O", "META.O", "ASML.O", "MRNA.O", "NFLX.O",
            "AMD.O", "INTC.O", "ADBE.O", "CRM.N", "ORCL.N",
            "CSCO.O", "JPM.N", "V.N", "MA.N", "WMT.N"
        ],
        initial_cash=1000000.0,
        initial_theta=Theta(
            gross_exposure=0.85,
            max_w=0.20,
            turnover_cap=0.25,
            risk_mode="neutral",
            enter_th=0.02,
            exit_th=-0.10
        ),
        start_date=None,
        end_date=None,
        train_test_split_ratio=0.7,
        llm_model="deepseek-chat"
    )

