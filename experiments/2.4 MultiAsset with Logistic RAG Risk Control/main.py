"""
多资产交易策略示例（使用 LogisticAgent with RAG Risk Control）
使用 MultiAssetLogisticAgentWithRAGGate 对指定股票池中的股票分别使用独立的 LogisticAgentWithRAGGate 获取信号
然后使用 RAG Gate 进行风险控制，最后进行权重归一化
"""
import sys
from pathlib import Path
from typing import List, Optional

# 添加项目根目录到 Python 路径
project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from trader.agent.multiasset_logistic_with_rag_gate import MultiAssetLogisticAgentWithRAGGate
from trader.backtest.account import Account
from trader.backtest.market import Market
from trader.backtest.engine import BacktestEngine
from trader.logger import get_logger, log_separator, log_section

logger = get_logger(__name__)


def multi_asset_strategy(
    stock_codes: Optional[List[str]] = None,
    initial_cash: float = 1000000.0,
    max_position_weight: float = 0.4,  # 单个股票最大40%（增加投资力度，方便风险控制）
    min_score_threshold: float = 0.1,  # 提高阈值，只配置真正看好的股票（从0.0提高到0.1）
    max_total_weight: float = 1.0,  # 总仓位上限100%
    train_window_days: int = 252,
    prediction_horizon: int = 5,
    ret_threshold: float = 0.0,
    retrain_frequency: int = 20,
    train_test_split_ratio: float = 0.7,
    start_date: str = None,
    end_date: str = None,
    # 新增：交易频率控制参数
    min_trade_amount: float = 5000.0,  # 最小交易金额阈值（从100元提高到5000元）
    min_weight_change: float = 0.05,  # 最小权重变化阈值（5%），避免微小调整
    # RAG Gate 参数
    apply_rag_gate: bool = True,  # 是否应用 RAG Gate 风险控制
    llm_model: str = "deepseek-chat",  # LLM 模型名称
    llm_temperature: float = 0.3,  # LLM 温度参数（控制随机性）
    test_mode: bool = False,  # 测试模式，如果为 True，会打印更详细的调试信息
    test_force_reject: bool = False,  # 测试模式下的强制拒绝
):
    """
    多资产交易策略回测（带 RAG Gate 风险控制）
    
    Args:
        stock_codes: 股票代码列表，如果为 None 则使用默认股票池
        initial_cash: 初始资金
        max_position_weight: 单个股票最大配置比例
        min_score_threshold: 最小 score 阈值
        max_total_weight: 总配置比例上限
        train_window_days: LogisticAgent 训练窗口大小
        prediction_horizon: LogisticAgent 预测周期
        ret_threshold: LogisticAgent 收益阈值
        retrain_frequency: LogisticAgent 重新训练频率
        train_test_split_ratio: LogisticAgent 训练/测试分割比例
        start_date: 开始日期
        end_date: 结束日期
        min_trade_amount: 最小交易金额阈值
        min_weight_change: 最小权重变化阈值
        apply_rag_gate: 是否应用 RAG Gate 风险控制
        llm_model: LLM 模型名称
        llm_temperature: LLM 温度参数
        test_mode: 测试模式
        test_force_reject: 测试模式下的强制拒绝
    """
    for line in log_section("多资产交易策略回测（Logistic with RAG Risk Control）"):
        logger.info(line)
    
    # 初始化市场、账户和回测引擎
    market = Market(price_adjustment=0.01)
    account = Account(initial_cash=initial_cash)
    
    # 如果没有指定股票代码，使用默认股票池
    if stock_codes is None:
        stock_codes = ["AAPL.O", "AMZN.O", "ASML.O", "META.O", "MRNA.O"]
    
    # 验证股票代码是否在数据库中存在
    all_symbols = market.get_all_symbols()
    valid_stock_codes = []
    for code in stock_codes:
        if code in all_symbols:
            valid_stock_codes.append(code)
        else:
            logger.warning(f"股票 {code} 不在数据库中，跳过")
    
    if not valid_stock_codes:
        logger.error("没有有效的股票代码")
        return
    
    logger.info(f"使用股票: {valid_stock_codes}")
    logger.info(f"初始资金: {initial_cash:,.2f} 元")
    logger.info(f"单股最大仓位: {max_position_weight*100:.0f}%")
    logger.info(f"总仓位上限: {max_total_weight*100:.0f}%")
    logger.info(f"最小 score 阈值: {min_score_threshold:.2f}")
    logger.info(f"最小交易金额: {min_trade_amount:,.0f} 元")
    logger.info(f"最小权重变化阈值: {min_weight_change*100:.0f}%")
    
    # RAG Gate 参数
    if apply_rag_gate:
        logger.info(f"RAG Gate: 启用")
        logger.info(f"  - LLM 模型: {llm_model}")
        logger.info(f"  - LLM 温度: {llm_temperature:.2f}")
        logger.info(f"  - 测试模式: {test_mode}")
        logger.info(f"  - RAG 系统会检索历史数据、新闻、趋势等多维度信息")
        logger.info(f"  - 基于检索到的证据进行风险控制决策")
    else:
        logger.info(f"RAG Gate: 禁用")
    
    # 生成报告标题（与实验名称对齐）
    report_title = "2.4 MultiAsset with Logistic RAG Risk Control"
    engine = BacktestEngine(
        account, market,
        report_title=report_title,
        train_test_split_ratio=train_test_split_ratio
    )
    
    # 创建多资产交易代理（使用 LogisticAgentWithRAGGate）
    agent = MultiAssetLogisticAgentWithRAGGate(
        stock_codes=valid_stock_codes,
        name="MultiAsset_Logistic_RAGGate",
        max_position_weight=max_position_weight,
        min_score_threshold=min_score_threshold,
        max_total_weight=max_total_weight,
        train_window_days=train_window_days,
        prediction_horizon=prediction_horizon,
        ret_threshold=ret_threshold,
        retrain_frequency=retrain_frequency,
        train_test_split_ratio=train_test_split_ratio,
        llm_model=llm_model,
        llm_temperature=llm_temperature,
        test_mode=test_mode,
        test_force_reject=test_force_reject,
        use_parallel=False,  # 禁用并行计算
        max_workers=None
    )
    
    # 记录上一日的权重，用于计算权重变化
    last_weights = {}
    
    # 注册交易日回调：执行策略
    def on_trading_day(eng: BacktestEngine, date: str):
        """每个交易日的回调"""
        nonlocal last_weights  # 允许修改外部变量
        
        # 检查所有股票是否都有价格数据（避免非交易日或数据缺失）
        market_prices = eng.get_market_prices(valid_stock_codes)
        if len(market_prices) < len(valid_stock_codes):
            # 有股票缺少价格数据，跳过该日期
            missing_stocks = [code for code in valid_stock_codes if code not in market_prices]
            logger.debug(f"[{date}] 跳过：以下股票缺少价格数据: {missing_stocks}")
            return
        
        # 更新 agent 状态
        agent.on_date(eng, date)
        
        # 获取所有股票的归一化权重（应用 RAG Gate）
        weights = agent.get_all_weights(
            eng,
            apply_rag_gate=apply_rag_gate
        )
        
        # 获取账户权益
        account_equity = account.equity(market_prices)
        
        # 对每支股票执行交易
        for stock_code in valid_stock_codes:
            new_weight = weights.get(stock_code, 0.0)
            old_weight = last_weights.get(stock_code, 0.0)
            
            # 检查权重变化是否足够大（避免频繁微调）
            weight_change = abs(new_weight - old_weight)
            if weight_change < min_weight_change:
                # 权重变化太小，跳过交易
                continue
            
            # 计算目标持仓金额
            target_value = account_equity * new_weight
            
            # 获取当前持仓
            position = account.get_position(stock_code)
            current_value = 0.0
            current_price = eng.get_current_price(stock_code)
            if position and current_price:
                current_value = position['shares'] * current_price
            
            # 计算需要调整的金额
            diff_value = target_value - current_value
            
            # 执行交易（提高最小交易金额阈值，减少频繁交易）
            if abs(diff_value) > min_trade_amount:
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
        
        # 更新上一日的权重
        last_weights = weights.copy()
    
    engine.on_date(on_trading_day)
    
    # 运行回测（使用第一支股票的日期范围）
    if valid_stock_codes:
        logger.info("")
        logger.info("开始回测...")
        engine.run(valid_stock_codes[0], start_date=start_date, end_date=end_date)
        
        if engine.train_test_split_date:
            logger.info(f"训练/测试分割日期: {engine.train_test_split_date}")
        
        # 计算最终结果
        final_prices = {}
        for stock_code in valid_stock_codes:
            final_price = market.get_price(
                stock_code,
                end_date if end_date else market.get_available_dates(stock_code)[-1]
            )
            if final_price is None:
                final_price = market.get_price(stock_code)
            if final_price:
                final_prices[stock_code] = final_price
        
        if final_prices:
            equity = account.equity(final_prices)
            profit = account.get_total_profit(final_prices)
            return_pct = account.get_total_return(final_prices)
            
            # 输出结果
            logger.info("")
            for line in log_section("回测结果"):
                logger.info(line)
            logger.info(f"初始资金: {initial_cash:,.2f} 元")
            logger.info(f"最终权益: {equity:,.2f} 元")
            logger.info(f"总盈亏: {profit:+,.2f} 元")
            logger.info(f"总收益率: {return_pct:+.2f}%")
            logger.info(f"交易次数: {len(account.trades)}")
            logger.info(log_separator())
            
            # 输出详细账户摘要
            logger.info("")
            logger.info(account.summary(final_prices))
            
            # 输出 RAG Gate 统计信息
            if apply_rag_gate:
                logger.info("")
                for line in log_section("RAG Gate 统计信息"):
                    logger.info(line)
                gate_stats = agent.get_gate_stats()
                total_passed = 0
                total_skipped = 0
                for stock_code, stats in gate_stats.items():
                    passed = stats.get("gate_passed_count", 0)
                    skipped = stats.get("gate_skipped_count", 0)
                    total = stats.get("total_evaluations", 0)
                    total_passed += passed
                    total_skipped += skipped
                    if total > 0:
                        logger.info(
                            f"{stock_code}: 通过={passed}, 拒绝={skipped}, "
                            f"总计={total}, 拒绝率={skipped/total*100:.1f}%"
                        )
                total_evaluations = total_passed + total_skipped
                if total_evaluations > 0:
                    logger.info(
                        f"总计: 通过={total_passed}, 拒绝={total_skipped}, "
                        f"总计={total_evaluations}, 拒绝率={total_skipped/total_evaluations*100:.1f}%"
                    )
                logger.info(log_separator())
            
            # 生成多资产组合报告
            if engine.report:
                logger.info("")
                logger.info("生成多资产组合报告...")
                strategy_params = {
                    "max_position_weight": f"{max_position_weight*100:.0f}%",
                    "min_score_threshold": f"{min_score_threshold:.2f}",
                    "max_total_weight": f"{max_total_weight*100:.0f}%",
                    "train_window_days": train_window_days,
                    "prediction_horizon": prediction_horizon,
                    "ret_threshold": ret_threshold,
                    "retrain_frequency": retrain_frequency,
                    "min_trade_amount": f"{min_trade_amount:,.0f} 元",
                    "min_weight_change": f"{min_weight_change*100:.0f}%"
                }
                
                # 添加 RAG Gate 参数
                if apply_rag_gate:
                    strategy_params["apply_rag_gate"] = "启用"
                    strategy_params["llm_model"] = llm_model
                    strategy_params["llm_temperature"] = f"{llm_temperature:.2f}"
                    strategy_params["rag_description"] = "RAG 系统检索历史数据、新闻、趋势等多维度信息，基于证据进行风险控制"
                else:
                    strategy_params["apply_rag_gate"] = "禁用"
                
                report_file = engine.report.generate_multi_asset_report(
                    account=account,
                    stock_codes=valid_stock_codes,
                    start_date=start_date or engine.trading_dates[0] if engine.trading_dates else "N/A",
                    end_date=end_date or engine.trading_dates[-1] if engine.trading_dates else "N/A",
                    strategy_params=strategy_params
                )
                logger.info(f"多资产组合报告已保存: {report_file}")


if __name__ == "__main__":
    # 执行多资产交易策略（带 RAG Gate 风险控制）
    # 默认使用指定的5支股票：AAPL.O, AMZN.O, ASML.O, META.O, MRNA.O
    multi_asset_strategy(
        stock_codes=["AAPL.O", "AMZN.O", "ASML.O", "META.O", "MRNA.O"],
        initial_cash=1000000.0,
        max_position_weight=0.4,  # 单个股票最大40%
        min_score_threshold=0.05,  # 降低阈值到0.05，增加交易机会（使用平方映射，低score仓位仍然较小）
        max_total_weight=1.0,  # 总仓位上限100%
        train_window_days=252,
        prediction_horizon=5,
        ret_threshold=0.0,
        retrain_frequency=20,
        train_test_split_ratio=0.7,
        start_date=None,
        end_date=None,
        # 优化参数：减少频繁交易
        min_trade_amount=5000.0,  # 最小交易金额5000元
        min_weight_change=0.05,  # 权重变化至少5%才交易（避免微小调整）
        # RAG Gate 参数
        apply_rag_gate=True,  # 启用 RAG Gate 风险控制
        llm_model="deepseek-chat",  # LLM 模型名称
        llm_temperature=0.3,  # LLM 温度参数（控制随机性）
        test_mode=False,  # 测试模式
        test_force_reject=False,  # 测试模式下的强制拒绝
    )

