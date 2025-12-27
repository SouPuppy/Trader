"""
Logistic Regression 逻辑回归策略回测示例（带 RAG 风险控制）
使用 LogisticAgentWithRAGGate 实现基于逻辑回归的预测策略，增加投资比例并通过 RAG Gate 进行风险控制

参数设置：
- max_position_weight=0.5 (50%) - 单个股票最大仓位（增加投资力度，方便风险控制）
- max_total_weight=1.0 (100%) - 总仓位上限
- RAG Gate: 在执行交易前，使用 RAG 系统检索历史数据、新闻、趋势等信息，基于证据评估是否应该执行

与 1.3.3 的区别：
- 1.3.3 使用简单的 LLM Gate（直接调用 LLM，没有检索历史数据）
- 1.3.4 使用 RAG Gate（检索历史数据、新闻、趋势等，基于证据进行决策）
- RAG Gate 能够利用历史数据、新闻、趋势等多维度信息，提供更准确的风险控制
"""
import sys
from pathlib import Path

# 添加项目根目录到 Python 路径
project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from trader.agent.agent_logistic_with_rag_gate import LogisticAgentWithRAGGate
from trader.backtest.account import Account
from trader.backtest.market import Market
from trader.backtest.engine import BacktestEngine
from trader.backtest.report import BacktestReport
from trader.logger import get_logger, log_separator, log_section

logger = get_logger(__name__)


def logistic_strategy_with_rag_gate(
    stock_code: str = "AVGO.O",
    initial_cash: float = 1000000.0,
    feature_names: list = None,
    train_window_days: int = 252,  # 约1年交易日
    prediction_horizon: int = 5,  # 预测未来5天
    ret_threshold: float = 0.0,  # 收益阈值
    retrain_frequency: int = 20,  # 每20个交易日重新训练
    max_position_weight: float = 0.5,  # 单个股票最大50%（增加投资力度，方便风险控制）
    min_score_threshold: float = 0.0,  # score >= 0 才买入
    max_total_weight: float = 1.0,  # 总仓位不超过100%（增加投资比例）
    train_test_split_ratio: float = 0.7,  # 训练/测试分割比例（70%用于训练）
    llm_model: str = "deepseek-chat",
    llm_temperature: float = 0.3,
    start_date: str = None,
    end_date: str = None
):
    """
    逻辑回归策略回测（使用 LogisticAgentWithRAGGate，增加投资比例 + RAG Gate 风险控制）
    
    Args:
        stock_code: 股票代码
        initial_cash: 初始资金
        feature_names: 使用的特征名称列表，如果为 None 则使用默认特征
        train_window_days: 训练窗口大小（交易日数）
        prediction_horizon: 预测未来多少天的收益
        ret_threshold: 收益阈值，未来收益 > threshold 为正样本
        retrain_frequency: 重新训练频率（每N个交易日）
        max_position_weight: 单个股票最大配置比例（增加到50%，方便风险控制）
        min_score_threshold: 最小 score 阈值
        max_total_weight: 总配置比例上限（增加到100%）
        train_test_split_ratio: 训练/测试分割比例（默认0.7，即70%用于训练）
        llm_model: LLM 模型名称（RAG 系统内部使用）
        llm_temperature: LLM 温度参数（控制随机性）
        start_date: 开始日期
        end_date: 结束日期
    """
    for line in log_section("逻辑回归策略回测（增加投资比例 + RAG Gate 风险控制）"):
        logger.info(line)
    logger.info(f"股票代码: {stock_code}")
    logger.info(f"初始资金: {initial_cash:,.2f} 元")
    logger.info(f"训练窗口: {train_window_days} 天")
    logger.info(f"预测周期: {prediction_horizon} 天")
    logger.info(f"收益阈值: {ret_threshold:.2%}")
    logger.info(f"重新训练频率: 每 {retrain_frequency} 个交易日")
    logger.info(f"单股最大仓位: {max_position_weight*100:.0f}%")
    logger.info(f"总仓位上限: {max_total_weight*100:.0f}%")
    logger.info(f"RAG 系统 LLM 模型: {llm_model}")
    logger.info(f"RAG 系统 LLM 温度: {llm_temperature}")
    logger.info(f"训练/测试分割比例: {train_test_split_ratio*100:.0f}% / {(1-train_test_split_ratio)*100:.0f}%")
    logger.info("")
    logger.info("RAG Gate 说明:")
    logger.info("  - RAG 系统会检索历史数据、新闻、趋势等多维度信息")
    logger.info("  - 基于检索到的证据进行风险控制决策")
    logger.info("  - 相比简单的 LLM Gate，RAG Gate 能够利用更多历史信息")
    
    # 初始化市场、账户和回测引擎
    market = Market(price_adjustment=0.01)
    account = Account(initial_cash=initial_cash)
    
    # 设置输出目录为实验文件夹
    experiment_dir = Path(__file__).parent.name
    from trader.config import PROJECT_ROOT
    output_dir = PROJECT_ROOT / 'output' / 'backtest' / experiment_dir
    
    # 不生成自动报告，只记录 daily_records
    engine = BacktestEngine(
        account, market, 
        report_title=None,  # 不自动生成报告
        report_output_dir=output_dir,
        train_test_split_ratio=train_test_split_ratio
    )
    
    # 获取可用日期
    available_dates = market.get_available_dates(stock_code)
    if not available_dates:
        logger.error(f"未找到股票 {stock_code} 的数据")
        return
    
    logger.info(f"数据范围: {available_dates[0]} 至 {available_dates[-1]}")
    
    # 创建带 RAG Gate 的逻辑回归 Agent
    agent = LogisticAgentWithRAGGate(
        name="Logistic_RAGGate_Strategy",
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
        llm_temperature=llm_temperature
    )
    
    # 注册交易日回调：执行策略
    def on_trading_day(eng: BacktestEngine, date: str):
        """每个交易日的回调"""
        # 更新 agent 状态（更新交易日计数）
        agent.on_date(eng, date)
        
        # 获取当前股票的 score
        score = agent.score(stock_code, eng)
        
        # 计算权重
        weight = agent.weight(stock_code, score, eng)
        
        # 如果权重为0，不需要交易
        if weight == 0.0:
            # 检查是否需要卖出（如果当前有持仓但权重为0）
            position = account.get_position(stock_code)
            if position and position['shares'] > 0:
                # 卖出所有持仓
                eng.sell(stock_code, shares=position['shares'])
            return
        
        # 获取账户权益
        market_prices = eng.get_market_prices([stock_code])
        account_equity = account.equity(market_prices)
        
        # 计算目标持仓金额
        target_value = account_equity * weight
        
        # 获取当前持仓
        position = account.get_position(stock_code)
        current_value = 0.0
        current_price = eng.get_current_price(stock_code)
        if position and current_price:
            current_value = position['shares'] * current_price
        
        # 计算需要调整的金额
        diff_value = target_value - current_value
        
        # 如果变化很小，不需要交易
        if abs(diff_value) < 100:
            return
        
        # 使用 RAG Gate 评估是否应该执行交易
        logger.info(f"[{agent.name}] 开始 RAG Gate 评估: {stock_code} on {date}, score={score:.4f}, weight={weight:.2%}")
        gate_result = agent._evaluate_gate(stock_code, eng, date, score, weight)
        
        should_execute = gate_result.get("should_execute", True)
        reasoning = gate_result.get("reasoning", "")
        
        if not should_execute:
            # Gate 拒绝执行，跳过本次交易
            agent.gate_skipped_count += 1
            logger.info(
                f"[{agent.name}] RAG Gate 拒绝执行交易: {stock_code} on {date}, "
                f"score={score:.4f}, weight={weight:.2%}, "
                f"原因: {reasoning[:100]}..."
            )
            return
        
        # Gate 通过，执行交易
        agent.gate_passed_count += 1
        logger.info(
            f"[{agent.name}] RAG Gate 通过，执行交易: {stock_code} on {date}, "
            f"score={score:.4f}, weight={weight:.2%}, "
            f"原因: {reasoning[:100]}..."
        )
        
        # 执行交易（diff_value 已经在前面检查过，这里直接执行）
        if diff_value > 0:
            # 买入
            eng.buy(stock_code, amount=diff_value)
        else:
            # 卖出
            shares_to_sell = int(abs(diff_value) / current_price) if current_price else 0
            if shares_to_sell > 0 and position:
                # 不能卖出超过持仓数量
                shares_to_sell = min(shares_to_sell, position['shares'])
                if shares_to_sell > 0:
                    eng.sell(stock_code, shares=shares_to_sell)
    
    engine.on_date(on_trading_day)
    
    # 运行回测
    logger.info("")
    logger.info("开始回测...")
    engine.run(stock_code, start_date=start_date, end_date=end_date)
    
    # 训练/测试分割日期已由 BacktestEngine 统一管理并设置到报告中
    if engine.train_test_split_date:
        logger.info(f"训练/测试分割日期: {engine.train_test_split_date}")
    
    # 计算最终结果
    final_date = engine.current_date if engine.current_date else None
    if not final_date:
        final_date = end_date if end_date else available_dates[-1]
    
    final_price = market.get_price(stock_code, final_date)
    if final_price is None:
        final_price = market.get_price(stock_code)
    
    if final_price is None:
        logger.error("无法获取最终价格")
        return
    
    market_prices = {stock_code: final_price}
    equity = account.equity(market_prices)
    profit = account.get_total_profit(market_prices)
    return_pct = account.get_total_return(market_prices)
    
    # 获取 Gate 统计信息
    gate_stats = agent.get_gate_stats()
    
    # 输出结果
    logger.info("")
    for line in log_section("回测结果"):
        logger.info(line)
    logger.info(f"初始资金: {initial_cash:,.2f} 元")
    logger.info(f"最终权益: {equity:,.2f} 元")
    logger.info(f"总盈亏: {profit:+,.2f} 元")
    logger.info(f"总收益率: {return_pct:+.2f}%")
    logger.info(f"交易次数: {len(account.trades)}")
    logger.info(f"模型训练次数: {agent.train_count}")
    logger.info(f"RAG Gate 通过次数: {gate_stats['gate_passed_count']}")
    logger.info(f"RAG Gate 跳过次数: {gate_stats['gate_skipped_count']}")
    logger.info(f"RAG Gate 总评估次数: {gate_stats['total_evaluations']}")
    logger.info(log_separator())
    
    # 输出详细账户摘要
    logger.info("")
    logger.info(account.summary(market_prices))
    
    # 生成单股票报告
    logger.info("")
    logger.info("生成回测报告...")
    report = BacktestReport(output_dir=output_dir, title=None)
    report.daily_records = engine.report.daily_records if engine.report else []
    report.train_test_split_date = engine.train_test_split_date
    
    # 创建 assets 文件夹
    assets_dir = output_dir / 'assets'
    assets_dir.mkdir(exist_ok=True)
    
    actual_start_date = start_date or available_dates[0]
    actual_end_date = end_date or available_dates[-1]
    
    # 生成图表（保存到 assets 文件夹）
    original_output_dir = report.output_dir
    report.output_dir = assets_dir
    chart_file = report._generate_charts(stock_code, actual_start_date, actual_end_date)
    report.output_dir = original_output_dir
    
    # 重命名图表为标准格式
    chart_name = f"{stock_code.replace('.', '_')}_chart.png"
    new_chart_path = assets_dir / chart_name
    if chart_file and chart_file.exists() and chart_file != new_chart_path:
        if new_chart_path.exists():
            new_chart_path.unlink()
        chart_file.rename(new_chart_path)
        chart_file = new_chart_path
    
    # 生成报告
    report_file = report.generate_report(
        account, stock_code, actual_start_date, actual_end_date
    )
    
    # 更新报告中的图表路径为 assets/xxx_chart.png
    if chart_file and chart_file.exists() and report_file.exists():
        chart_relative_path = f"assets/{chart_name}"
        with open(report_file, 'r', encoding='utf-8') as f:
            content = f.read()
        import re
        # 替换图表路径
        content = re.sub(r'!\[回测走势图\]\([^)]+\)', f'![回测走势图]({chart_relative_path})', content)
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(content)
    
    # 重命名为 report.md
    final_report_file = output_dir / "report.md"
    if report_file != final_report_file:
        if final_report_file.exists():
            final_report_file.unlink()
        report_file.rename(final_report_file)
    
    # 删除 JSON 报告（不需要）
    json_files = list(output_dir.glob("backtest_report_*.json"))
    for json_file in json_files:
        json_file.unlink()
    
    logger.info(f"报告已保存: {final_report_file}")


if __name__ == "__main__":
    # 执行逻辑回归策略（增加投资比例 + RAG Gate 风险控制）- 只测试 AAPL.O
    logistic_strategy_with_rag_gate(
        stock_code="AAPL.O",
        initial_cash=1000000.0,
        feature_names=None,  # 使用默认特征
        train_window_days=252,  # 约1年交易日
        prediction_horizon=5,  # 预测未来5天
        ret_threshold=0.0,  # 收益阈值
        retrain_frequency=20,  # 每20个交易日重新训练
        max_position_weight=0.5,  # 单个股票最大50%（增加投资力度，方便风险控制）
        min_score_threshold=0.0,  # score >= 0 才买入
        max_total_weight=1.0,  # 总仓位不超过100%（从80%增加到100%）
        train_test_split_ratio=0.7,  # 训练/测试分割比例（70%用于训练）
        llm_model="deepseek-chat",
        llm_temperature=0.3,  # 较低温度以获得更一致的结果
        start_date=None,  # 从最早日期开始
        end_date=None     # 到最新日期
    )




