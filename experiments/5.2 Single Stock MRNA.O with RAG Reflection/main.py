"""
实验 5.2：单股票交易系统（带 RAG 反思层）- MRNA.O

Layer A: Signal Layer（信号层）
Layer B: Allocation & RiskControl Layer（分配与风控层）
Layer C: RAG 反思层（参数θ每周调整，使用 RAG 系统）

在单股票 MRNA.O 的基础上，使用 RAG 系统进行反思：
1. 评估过去一周的表现（周收益率、交易情况、市场状态）
2. 使用 RAG 系统检索历史数据、新闻、趋势等多维度信息
3. 基于 RAG 检索的证据生成参数调整建议
4. 解析 RAG 答案并应用参数调整

优化内容（继承自 5.1）：
1. 实现新闻信号功能：使用最基础的features（news_count），朴素算法，不做复杂分析
2. 改进趋势信号：使用多时间框架（1日、5日、20日）加权组合，捕捉趋势持续性
3. 优化基本面信号：综合PE、PB、PS多个估值指标，取平均值
4. 改进波动率信号：使用相对波动率（20日/60日），而非绝对波动率
5. 调整信号权重：提高趋势权重至0.5，降低波动率和新闻权重至0.15
6. 优化策略参数：
   - gross_exposure: 0.85（充分利用资金）
   - max_w: 0.20（单票上限，单股票场景下实际就是总仓位）
   - turnover_cap: 0.25（适度换手率）
   - enter_th: 0.02（进场阈值）
   - exit_th: -0.10（出场阈值）
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
from trader.agent.learnable_reflection import LearnableReflectionSystem
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
    stock_code: str,
    reflection_id: int
) -> str:
    """
    构建 RAG 反思问题（单股票版本）
    
    Args:
        current_theta: 当前参数θ
        weekly_return: 周收益率（小数）
        account_equity: 当前账户权益
        initial_cash: 初始资金
        stock_code: 股票代码
        reflection_id: 反思轮次ID
        
    Returns:
        str: RAG 问题文本
    """
    total_return = (account_equity - initial_cash) / initial_cash if initial_cash > 0 else 0.0
    
    question = f"""作为专业的量化交易策略分析师，请基于以下信息评估单股票交易策略的表现，并提供参数调整建议。

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
- 交易股票: {stock_code}

【分析要求】
请基于 RAG 系统检索到的历史数据、新闻、趋势等多维度信息，分析：
1. 本周表现评估：本周收益率 {weekly_return*100:.2f}% 是否良好？与市场整体表现相比如何？
   **重要**：周收益率是评估策略表现的核心指标，必须优先考虑
2. 市场环境分析：当前市场环境如何？{stock_code} 有哪些风险或机会？
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
- **重要**：分析单股票 {stock_code} 的表现
- **重要**：周收益率是评估策略表现的核心指标，必须优先考虑周收益率进行参数调整
- 参数范围限制：
  * gross_exposure: [0.4, 0.85]（上限与 5.1 保持一致）
  * max_w: [0.08, 0.20]（上限与 5.1 保持一致）
  * turnover_cap: [0.10, 0.30]（上限与 5.1 保持一致）
  * enter_th: [0.01, 0.05]
  * exit_th: [-0.15, -0.08]
- 如果某个参数不需要调整，必须设置为 null（不要使用当前值）
- **调整原则（与 5.1 的 naive reflection 保持一致）**：
  * 如果周收益率 > 1.5%，适度提高仓位（gross_exposure, max_w）或降低进场阈值（enter_th），调整幅度约 2%
  * 如果周收益率 < -1.5%，适度降低仓位或提高进场阈值，调整幅度约 2%
  * 如果周收益率在 ±1.5% 范围内，保持参数不变（不调整）
  * **关键**：只在周收益率明显偏离 ±1.5% 时才调整，避免频繁调整
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
        
        # 参数范围校验（与 5.1 保持一致）
        new_theta_dict["gross_exposure"] = max(0.4, min(0.85, new_theta_dict["gross_exposure"]))
        new_theta_dict["max_w"] = max(0.08, min(0.20, new_theta_dict["max_w"]))  # 上限与 5.1 保持一致
        new_theta_dict["turnover_cap"] = max(0.10, min(0.30, new_theta_dict["turnover_cap"]))  # 上限与 5.1 保持一致
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
    stock_code: str,
    reflection_id: int,
    decision_time: str,
    llm_model: str = "deepseek-chat"
) -> Theta:
    """
    RAG 反思：使用 RAG 系统生成参数调整建议（单股票版本）
    
    Args:
        current_theta: 当前参数θ
        weekly_return: 周收益率（小数）
        account_equity: 当前账户权益
        initial_cash: 初始资金
        stock_code: 股票代码
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
        return naive_reflection_fallback(current_theta, weekly_return, reflection_id)
    
    # 构建反思问题
    question = build_reflection_question(
        current_theta=current_theta,
        weekly_return=weekly_return,
        account_equity=account_equity,
        initial_cash=initial_cash,
        stock_code=stock_code,
        reflection_id=reflection_id
    )
    
    logger.info(f"[RAG Reflection] 开始 RAG 反思（第 {reflection_id} 次）")
    logger.info(f"[RAG Reflection] 周收益率: {weekly_return*100:.2f}%")
    logger.info(f"[RAG Reflection] 交易股票: {stock_code}")
    
    try:
        logger.info(f"[RAG Reflection] 调用 RAG 系统，使用股票代码: {stock_code}")
        verified_answer = rag_answer(
            question=question,
            stock_code=stock_code,
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
        
        # 如果 degraded mode、证据不足或拒绝调整，直接使用保守的 naive reflection
        if is_degraded or has_insufficient_evidence or not adjustments or rag_refused_adjustment:
            logger.warning(f"[RAG Reflection] RAG 系统状态: degraded={is_degraded}, insufficient_evidence={has_insufficient_evidence}, refused={rag_refused_adjustment}")
            logger.info("[RAG Reflection] RAG 系统不可用或拒绝调整，使用保守的 naive reflection（与 5.1 保持一致）")
            
            # 直接使用保守的 naive reflection（与 5.1 保持一致）
            naive_theta = naive_reflection_fallback(current_theta, weekly_return, reflection_id)
            logger.info(f"[RAG Reflection] 使用 naive reflection 最终参数: {current_theta} -> {naive_theta}")
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
    保守的反思回退方案：与 5.1 的 naive reflection 保持一致，避免过度激进
    
    Args:
        current_theta: 当前参数θ
        weekly_return: 周收益率（小数）
        reflection_id: 反思轮次ID
        
    Returns:
        Theta: 调整后的参数θ
    """
    new_theta = current_theta.copy()
    new_theta.reflection_id = reflection_id
    
    # 保守的调整策略：与 5.1 保持一致
    # 1. 大幅提高阈值：只在周收益率 > 1.5% 或 < -1.5% 时才调整，减少频繁调整
    positive_threshold = 0.015  # 1.5%（大幅提高，减少调整频率）
    negative_threshold = -0.015  # -1.5%
    
    # 2. 降低调整幅度：2%，使调整更平滑
    adjustment_factor = 0.02  # 2% 的调整幅度（更保守）
    
    if weekly_return > positive_threshold:  # 周收益率 > 1.5%，表现明显好
        # 稍微激进：增加仓位、降低阈值
        logger.info(f"[Naive Reflection Fallback] 周收益率 {weekly_return*100:.2f}% > {positive_threshold*100:.2f}%，表现良好，适度激进调整")
        
        # 增加总仓位（但不超过上限，且更保守的增长）
        new_theta.gross_exposure = min(0.85, current_theta.gross_exposure * (1 + adjustment_factor))
        
        # 增加单票上限（但不超过上限）
        new_theta.max_w = min(0.20, current_theta.max_w * (1 + adjustment_factor))
        
        # 适度增加换手上限（允许更多交易，但更保守）
        new_theta.turnover_cap = min(0.30, current_theta.turnover_cap * (1 + adjustment_factor * 0.6))
        
        # 风险模式：保持中性，不轻易切换
        if current_theta.risk_mode == "risk_off":
            new_theta.risk_mode = "neutral"
        
        # 降低进场阈值（更容易进场，但保持合理下限）
        new_theta.enter_th = max(0.01, current_theta.enter_th * (1 - adjustment_factor * 0.8))
        
        # 适度放宽出场阈值（更不容易止损）
        new_theta.exit_th = min(-0.08, current_theta.exit_th * (1 - adjustment_factor * 0.6))
        
    elif weekly_return < negative_threshold:  # 周收益率 < -1.5%，表现明显差
        # 更保守：降低仓位、提高阈值
        logger.info(f"[Naive Reflection Fallback] 周收益率 {weekly_return*100:.2f}% < {negative_threshold*100:.2f}%，表现不佳，更保守调整")
        
        # 降低总仓位（但不低于下限）
        new_theta.gross_exposure = max(0.50, current_theta.gross_exposure * (1 - adjustment_factor))
        
        # 降低单票上限（但不低于下限）
        new_theta.max_w = max(0.10, current_theta.max_w * (1 - adjustment_factor))
        
        # 降低换手上限（减少交易，降低交易成本）
        new_theta.turnover_cap = max(0.15, current_theta.turnover_cap * (1 - adjustment_factor))
        
        # 风险模式：更保守的切换，降低风险偏好
        if current_theta.risk_mode == "risk_on":
            new_theta.risk_mode = "neutral"
        elif current_theta.risk_mode == "neutral":
            new_theta.risk_mode = "risk_off"
        
        # 提高进场阈值（更难进场）
        new_theta.enter_th = min(0.04, current_theta.enter_th * (1 + adjustment_factor * 0.8))
        
        # 适度收紧出场阈值（更容易止损，但不过度）
        new_theta.exit_th = max(-0.12, current_theta.exit_th * (1 - adjustment_factor * 0.6))
        
    else:
        # 在 ±1.5% 范围内，保持参数不变（大幅减少调整频率）
        logger.info(f"[Naive Reflection Fallback] 周收益率 {weekly_return*100:.2f}% 在正常波动范围（±{positive_threshold*100:.2f}%），保持参数不变")
    
    logger.info(f"[Naive Reflection Fallback] 参数调整: {current_theta} -> {new_theta}")
    return new_theta


def single_stock_strategy_with_rag_reflection(
    stock_code: str = "MRNA.O",
    initial_cash: float = 1000000.0,
    initial_theta: Theta = None,
    start_date: str = None,
    end_date: str = None,
    train_test_split_ratio: float = 0.7,
    llm_model: str = "deepseek-chat"
):
    """
    单股票交易策略（带 RAG 反思层）
    
    Args:
        stock_code: 股票代码
        initial_cash: 初始资金
        initial_theta: 初始参数θ（如果为None则使用默认值）
        start_date: 开始日期
        end_date: 结束日期
        train_test_split_ratio: 训练/测试分割比例
        llm_model: LLM 模型名称（RAG 系统内部使用）
    """
    if initial_theta is None:
        initial_theta = Theta(
            gross_exposure=0.85,  # 与 5.1 保持一致，提高总仓位到85%，充分利用资金
            max_w=0.20,  # 与 5.1 保持一致，单票上限20%，单股票场景下实际就是总仓位
            turnover_cap=0.25,  # 与 5.1 保持一致，适度换手率上限
            risk_mode="neutral",
            enter_th=0.02,  # 与 5.1 保持一致，降低进场阈值，让更多股票有机会
            exit_th=-0.10  # 放宽出场阈值，避免过早止损
        )
    
    for line in log_section("单股票交易系统（带 RAG 反思层）"):
        logger.info(line)
    logger.info(f"股票代码: {stock_code}")
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
    available_dates = market.get_available_dates(stock_code)
    if not available_dates:
        logger.error(f"未找到股票数据")
        return
    
    logger.info(f"数据范围: {available_dates[0]} 至 {available_dates[-1]}")
    
    # 创建Layer A: Signal Layer（优化权重，与 5.1 保持一致）
    signal_layer = SignalLayer(
        name="SignalLayer",
        w_T=0.5,  # Trend权重（提高，趋势最重要），与 5.1 保持一致
        w_V=0.15,  # Volatility权重（降低），与 5.1 保持一致
        w_F=0.20,  # Fundamental权重（保持），与 5.1 保持一致
        w_N=0.15,  # News权重（降低，因为可能不稳定），与 5.1 保持一致
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
    
    # 初始化可训练的反思系统
    history_file = output_dir / 'reflection_history.json'
    learnable_reflection = LearnableReflectionSystem(history_file=history_file)
    
    # 跟踪待评估的反思（reflection_id -> (date, weekly_return_before, account_equity_before)）
    pending_evaluations: Dict[int, tuple] = {}
    
    # 注册交易日回调
    def on_trading_day(eng: ReflectionEngine, date: str):
        """每个交易日的回调"""
        nonlocal last_reflection_date, reflection_count
        
        # 记录每日参数（即使没有变化）
        eng.record_daily_theta(date)
        
        # 获取账户权益
        current_price = eng.get_current_price(stock_code)
        if current_price:
            market_prices = {stock_code: current_price}
        else:
            market_prices = {}
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
                if last_reflection_date:
                    try:
                        last_price = eng.get_price(stock_code, last_reflection_date)
                        current_price = eng.get_price(stock_code, date)
                        if last_price and current_price and last_price > 0:
                            weekly_return = (current_price - last_price) / last_price
                            logger.info(f"[Reflection] 计算周收益率（市场表现）: {last_reflection_date} -> {date}")
                            logger.info(f"[Reflection] 周收益率: {weekly_return*100:.2f}%")
                        else:
                            logger.warning(f"[Reflection] 无法计算市场收益率，跳过本次反思")
                            return
                    except Exception as e:
                        logger.warning(f"[Reflection] 计算市场收益率时出错: {e}")
                        return
                else:
                    # 第一次反思，使用第一周的市场表现
                    if available_dates and len(available_dates) > 5:
                        first_date = available_dates[0]
                        try:
                            first_price = eng.get_price(stock_code, first_date)
                            current_price = eng.get_price(stock_code, date)
                            if first_price and current_price and first_price > 0:
                                weekly_return = (current_price - first_price) / first_price
                                logger.info(f"[Reflection] 第一次反思，使用市场表现作为基准")
                                logger.info(f"[Reflection] 周收益率: {weekly_return*100:.2f}%")
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
            
            # 先尝试使用学习到的知识
            learned_theta = learnable_reflection.learn_adjustment(
                current_theta=current_theta,
                weekly_return=weekly_return,
                account_equity=account_equity
            )
            
            # 构建决策时间（ISO8601 格式）
            decision_time = f"{date}T00:00:00"
            
            # 如果学习系统有建议，优先使用；否则使用 RAG/naive
            if learned_theta:
                logger.info(f"[Learnable Reflection] 使用学习到的调整建议（第 {reflection_count} 次反思）")
                new_theta = learned_theta
                new_theta.reflection_id = reflection_count
                adjustment_reason = "基于历史经验学习到的调整"
                adjustment_source = "learned"
            else:
                # 使用 RAG 或 naive reflection
                new_theta = rag_reflection(
                    current_theta=current_theta,
                    weekly_return=weekly_return,
                    account_equity=account_equity,
                    initial_cash=account.initial_cash,
                    stock_code=stock_code,
                    reflection_id=reflection_count,
                    decision_time=decision_time,
                    llm_model=llm_model
                )
                adjustment_reason = "RAG/Naive reflection"
                adjustment_source = "rag"
            
            # 记录反思（在更新参数前记录调整前的状态）
            learnable_reflection.record_reflection(
                reflection_id=reflection_count,
                date=date,
                current_theta=current_theta,
                new_theta=new_theta,
                weekly_return_before=weekly_return,
                account_equity_before=account_equity,
                adjustment_reason=adjustment_reason,
                adjustment_source=adjustment_source
            )
            
            # 记录待评估的反思（一周后评估效果）
            pending_evaluations[reflection_count] = (date, weekly_return, account_equity)
            
            # 更新参数
            eng.update_theta(new_theta, date)
            
            # 更新 allocator 和 signal_layer 的参数
            allocator.update_theta(new_theta)
            signal_layer.update_theta(new_theta)
            
            # 评估之前的反思效果（如果已经过去一周）
            if last_reflection_date:
                # 找到一周前的反思ID
                for prev_reflection_id, (prev_date, prev_weekly_return, prev_equity) in list(pending_evaluations.items()):
                    # 计算日期差（简化：假设每周5个交易日）
                    try:
                        prev_date_obj = datetime.strptime(prev_date, "%Y-%m-%d")
                        current_date_obj = datetime.strptime(date, "%Y-%m-%d")
                        days_diff = (current_date_obj - prev_date_obj).days
                        
                        # 如果已经过去一周（5个交易日），评估效果
                        if days_diff >= 5:
                            # 计算调整后一周的收益率
                            if prev_date in weekly_equity_history:
                                prev_equity_at_reflection = weekly_equity_history[prev_date]
                                weekly_return_after = (account_equity - prev_equity_at_reflection) / prev_equity_at_reflection if prev_equity_at_reflection > 0 else 0.0
                                
                                # 更新反思结果
                                learnable_reflection.update_reflection_result(
                                    reflection_id=prev_reflection_id,
                                    weekly_return_after=weekly_return_after,
                                    account_equity_after=account_equity
                                )
                                
                                # 从待评估列表中移除
                                del pending_evaluations[prev_reflection_id]
                                
                                logger.info(f"[Learnable Reflection] 评估反思 {prev_reflection_id} 的效果: 调整前收益率 {prev_weekly_return*100:.2f}%, 调整后收益率 {weekly_return_after*100:.2f}%")
                    except Exception as e:
                        logger.debug(f"评估反思效果时出错: {e}")
            
            # 记录本次反思日期
            last_reflection_date = date
        
        # 获取目标权重（使用更新后的参数）
        target_weights = allocator.get_weights([stock_code], eng)
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
        
        # 如果变化很小（小于账户权益的0.5%），不需要交易（与 5.1 保持一致）
        min_trade_threshold = account_equity * 0.005
        if abs(diff_value) < min_trade_threshold:
            return
        
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
        # 如果现金比例过高，买入股票
        market_prices_after_trade = {stock_code: eng.get_current_price(stock_code)} if eng.get_current_price(stock_code) else {}
        account_equity_after_trade = account.equity(market_prices_after_trade)
        cash_ratio = account.cash / account_equity_after_trade if account_equity_after_trade > 0 else 1.0
        max_cash_ratio = 0.10  # 最多保留10%现金
        
        if cash_ratio > max_cash_ratio and account.cash > 0:
            # 计算需要投入的资金
            excess_cash = account.cash - (account_equity_after_trade * max_cash_ratio)
            min_trade_threshold_balance = account_equity_after_trade * 0.005
            if excess_cash > min_trade_threshold_balance:
                logger.info(f"[Balance] 现金比例 {cash_ratio*100:.2f}% 过高，需要投入 {excess_cash:,.2f} 元到股市")
                eng.buy(stock_code, amount=excess_cash)
                logger.debug(f"[Balance] 买入 {stock_code}: {excess_cash:,.2f} 元")
    
    engine.on_date(on_trading_day)
    
    # 运行回测
    logger.info("")
    logger.info("开始回测...")
    engine.run(stock_code, start_date=start_date, end_date=end_date)
    
    # 计算最终结果
    final_date = engine.current_date if engine.current_date else None
    if not final_date:
        final_date = end_date if end_date else available_dates[-1]
    
    final_price = market.get_price(stock_code, final_date)
    if final_price is None:
        final_price = market.get_price(stock_code)
    
    if not final_price:
        logger.error("无法获取最终价格")
        return
    
    market_prices = {stock_code: final_price}
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
    
    # 输出学习统计信息
    stats = learnable_reflection.get_statistics()
    logger.info("")
    logger.info("=== 可训练反思系统统计 ===")
    logger.info(f"总反思次数: {stats['total_reflections']}")
    logger.info(f"总评估次数: {stats['total_effectiveness_evaluations']}")
    logger.info(f"平均有效性评分: {stats['avg_effectiveness']:.3f}")
    logger.info(f"学习到的模式数: {stats['learned_patterns_count']}")
    for return_range, range_stats in stats['return_ranges'].items():
        logger.info(f"  {return_range}: {range_stats['count']} 次, 平均有效性 {range_stats['avg_effectiveness']:.3f}, 成功调整 {range_stats['successful_adjustments_count']} 次")
    
    logger.info(log_separator())
    
    # 生成参数化报告（单股票模式）
    logger.info("")
    logger.info("生成参数化报告...")
    actual_start_date = start_date or available_dates[0]
    actual_end_date = end_date or available_dates[-1]
    
    report_file = engine.generate_parametrized_report(
        stock_codes=[stock_code],
        start_date=actual_start_date,
        end_date=actual_end_date,
        strategy_name="单股票交易系统（带 RAG 反思层）",
        is_single_stock=True
    )
    
    logger.info(f"报告已保存: {report_file}")


if __name__ == "__main__":
    # 执行单股票交易策略（带 RAG 反思层）
    single_stock_strategy_with_rag_reflection(
        stock_code="MRNA.O",
        initial_cash=1000000.0,
        initial_theta=Theta(
            gross_exposure=0.85,  # 与 5.1 保持一致，提高总仓位到85%
            max_w=0.20,  # 与 5.1 保持一致，单票上限20%
            turnover_cap=0.25,  # 与 5.1 保持一致，适度换手率上限
            risk_mode="neutral",
            enter_th=0.02,  # 与 5.1 保持一致，降低进场阈值
            exit_th=-0.10
        ),
        start_date=None,
        end_date=None,
        train_test_split_ratio=0.7,
        llm_model="deepseek-chat"
    )

