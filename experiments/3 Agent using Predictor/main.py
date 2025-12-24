"""
Agent using Predictor - 使用 PredictorAgent 对单股票进行回测
对 model_config.toml 中 test stocks 的10只股票分别运行独立的backtest作为baseline
每只股票使用独立的账户和回测引擎，不是多资产组合策略
"""
import sys
from pathlib import Path
from typing import List, Optional

# 添加项目根目录到 Python 路径
project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from trader.agent.agent_predictor import PredictorAgent
from trader.backtest.account import Account
from trader.backtest.market import Market
from trader.backtest.engine import BacktestEngine
from trader.backtest.report import BacktestReport
from trader.predictor.config_loader import get_test_stocks, get_model_config
from trader.logger import get_logger, log_separator, log_section

logger = get_logger(__name__)


def predictor_strategy(
    stock_code: str,
    initial_cash: float = 1000000.0,
    max_position_weight: float = 0.1,
    min_score_threshold: float = 0.0,
    max_total_weight: float = 1.0,
    use_close_only: bool = True,
    seq_len: int = 21,
    use_square_weight: bool = False,
    start_date: str = None,
    end_date: str = None
):
    """
    使用 PredictorAgent 的单股票回测策略
    
    Args:
        stock_code: 股票代码
        initial_cash: 初始资金
        max_position_weight: 单个股票最大配置比例
        min_score_threshold: 最小 score 阈值
        max_total_weight: 总配置比例上限
        use_close_only: 是否只使用 close_price
        seq_len: 序列长度
        use_square_weight: 是否使用平方映射（让高score获得更多仓位）
        start_date: 开始日期
        end_date: 结束日期
    """
    for line in log_section(f"Predictor Agent 回测 - {stock_code}"):
        logger.info(line)
    
    # 初始化市场、账户和回测引擎（每只股票独立的账户）
    market = Market(price_adjustment=0.01)
    account = Account(initial_cash=initial_cash)
    
    # 验证股票代码是否在数据库中存在
    all_symbols = market.get_all_symbols()
    if stock_code not in all_symbols:
        logger.error(f"股票 {stock_code} 不在数据库中")
        return None
    
    logger.info(f"股票代码: {stock_code}")
    logger.info(f"初始资金: {initial_cash:,.2f} 元")
    logger.info(f"单股最大仓位: {max_position_weight*100:.0f}%")
    logger.info(f"总仓位上限: {max_total_weight*100:.0f}%")
    logger.info(f"最小 score 阈值: {min_score_threshold:.2f}")
    
    # 获取可用日期
    available_dates = market.get_available_dates(stock_code)
    if not available_dates:
        logger.error(f"未找到股票 {stock_code} 的数据")
        return None
    
    # 启用报告生成以记录每日状态，但不生成独立报告文件（通过设置 report_title=None）
    # Predictor策略使用全量数据，不进行训练/测试分割（train_test_split_ratio=0.0）
    engine = BacktestEngine(account, market, enable_report=True, report_title=None, train_test_split_ratio=0.0)
    
    # 创建 PredictorAgent（使用共享模型，对所有股票都适用）
    # 启用调试模式以输出详细的预测和交易决策信息
    agent = PredictorAgent(
        name=f"Predictor_{stock_code}",
        stock_code=stock_code,  # 已废弃，实际使用共享模型
        max_position_weight=max_position_weight,
        min_score_threshold=min_score_threshold,
        max_total_weight=max_total_weight,
        model_path=None,  # 使用默认共享模型路径 (weights/LSTM/model.pth)
        use_close_only=use_close_only,
        seq_len=seq_len,
        use_square_weight=use_square_weight,
        debug=True  # 启用详细调试信息
    )
    
    # 记录上一日的权重
    last_weight = 0.0
    
    # 执行交易（最小交易金额阈值）
    # 由于 Predictor 的预测收益率很小，产生的 weight 也较小
    # 降低阈值以允许小金额交易，但设置合理下限避免过度交易
    min_trade_amount = 10.0  # 降低到 10 元，允许小金额交易
    
    # 注册交易日回调：执行策略
    # 注意：把 min_trade_amount 作为默认参数绑定，避免闭包/作用域改动导致 UnboundLocalError
    def on_trading_day(eng: BacktestEngine, date: str, min_trade_amount: float = min_trade_amount):
        """每个交易日的回调"""
        nonlocal last_weight
        
        # 检查股票是否有价格数据
        current_price = eng.get_current_price(stock_code)
        if current_price is None:
            logger.debug(f"[{date}] 跳过：{stock_code} 缺少价格数据")
            return
        
        # 更新 agent 状态
        agent.on_date(eng, date)
        
        # 获取股票的 score
        score = agent.score(stock_code, eng)
        
        # 获取股票的 weight
        weight = agent.weight(stock_code, score, eng)
        
        # 归一化权重（确保不超过 max_total_weight）
        weights = {stock_code: weight}
        normalized_weights = agent.normalize_weights(weights)
        normalized_weight = normalized_weights.get(stock_code, 0.0)
        
        # 获取账户权益
        market_prices = {stock_code: current_price}
        account_equity = account.equity(market_prices)
        
        # 计算目标持仓金额
        target_value = account_equity * normalized_weight
        
        # 获取当前持仓
        position = account.get_position(stock_code)
        current_value = 0.0
        if position and current_price:
            current_value = position['shares'] * current_price
        
        # 计算需要调整的金额
        diff_value = target_value - current_value
        
        # 调试：记录前20个交易日的详细信息，以及所有交易决策
        if not hasattr(eng, '_debug_count'):
            eng._debug_count = 0
        eng._debug_count += 1
        
        # 详细调试信息（前20个交易日）
        if eng._debug_count <= 20:
            logger.info(
                f"[{date}] {stock_code} "
                f"score={score:.6f} weight={weight:.6f} normalized={normalized_weight:.6f} "
                f"target={target_value:.2f} current={current_value:.2f} diff={diff_value:.2f} "
                f"price={current_price:.2f}"
            )
        
        # 记录所有交易决策（无论是否执行）
        if abs(diff_value) > min_trade_amount:
            logger.info(
                f"[{date}] {stock_code} 交易信号: "
                f"score={score:.6f}, weight={weight:.6f}, normalized_weight={normalized_weight:.6f}, "
                f"diff_value={diff_value:.2f} (阈值={min_trade_amount:.2f})"
            )
        
        # 执行交易
        if abs(diff_value) > min_trade_amount:
            if diff_value > 0:
                # 买入
                eng.buy(stock_code, amount=diff_value)
                if eng._debug_count <= 10:
                    logger.info(f"[{date}] 买入 {stock_code} 金额={diff_value:.2f}")
            else:
                # 卖出
                shares_to_sell = int(abs(diff_value) / current_price) if current_price else 0
                if shares_to_sell > 0 and position:
                    shares_to_sell = min(shares_to_sell, position['shares'])
                    if shares_to_sell > 0:
                        eng.sell(stock_code, shares=shares_to_sell)
                        if eng._debug_count <= 10:
                            logger.info(f"[{date}] 卖出 {stock_code} 股数={shares_to_sell}")
        
        # 更新上一日的权重
        last_weight = normalized_weight
    
    engine.on_date(on_trading_day)
    
    # 运行回测
    logger.info("")
    logger.info("开始回测...")
    engine.run(stock_code, start_date=start_date, end_date=end_date)
    
    # 使用回测引擎实际运行的最后一个日期，或者最后一个可用日期
    if end_date and end_date in available_dates:
        final_date = end_date
    else:
        final_date = engine.current_date if engine.current_date else available_dates[-1]
    
    final_price = market.get_price(stock_code, final_date)
    if final_price is None:
        logger.error(f"Final price unavailable for date: {final_date}")
        return None
    
    market_prices = {stock_code: final_price}
    equity = account.equity(market_prices)
    profit = account.get_total_profit(market_prices)
    return_pct = account.get_total_return(market_prices)
    
    # 返回结果字典，包含用于生成报告的信息
    return {
        'stock_code': stock_code,
        'initial_cash': initial_cash,
        'final_equity': equity,
        'profit': profit,
        'return_pct': return_pct,
        'num_trades': len(account.trades),
        'data_range': f"{available_dates[0]} → {available_dates[-1]}",
        # 保存账户和引擎信息，用于生成图表和详细指标
        'account': account,
        'engine': engine,
        'market': market,
        'start_date': start_date or available_dates[0],
        'end_date': end_date or available_dates[-1]
    }


def run_all_stocks_baseline(
    stock_codes: Optional[List[str]] = None,
    initial_cash: float = 1000000.0,
    max_position_weight: float = 0.1,
    min_score_threshold: float = 0.0,
    max_total_weight: float = 1.0,
    use_close_only: bool = True,
    seq_len: int = 21,
    use_square_weight: bool = False,
    start_date: str = None,
    end_date: str = None
):
    """
    对所有股票分别运行独立的backtest作为baseline
    
    注意：这是对每只股票单独测试，每只股票使用独立的账户和回测引擎，
    不是多资产组合策略。
    
    Args:
        stock_codes: 股票代码列表，如果为 None 则从 model_config.toml 读取 test stocks
        其他参数同 predictor_strategy
    """
    # 如果没有指定股票代码，从配置文件读取
    if stock_codes is None:
        stock_codes = get_test_stocks()
    
    logger.info("")
    for line in log_section("Predictor Agent Baseline - 单股票回测"):
        logger.info(line)
    logger.info(f"股票数量: {len(stock_codes)}")
    logger.info(f"股票列表: {stock_codes}")
    logger.info("注意：每只股票使用独立的账户和回测引擎，分别测试")
    logger.info("")
    
    # 运行所有股票的回测
    results = []
    for i, stock_code in enumerate(stock_codes, 1):
        logger.info("")
        logger.info(f"========== [{i}/{len(stock_codes)}] {stock_code} ==========")
        
        try:
            result = predictor_strategy(
                stock_code=stock_code,
                initial_cash=initial_cash,
                max_position_weight=max_position_weight,
                min_score_threshold=min_score_threshold,
                max_total_weight=max_total_weight,
                use_close_only=use_close_only,
                seq_len=seq_len,
                use_square_weight=use_square_weight,
                start_date=start_date,
                end_date=end_date
            )
            
            if result:
                results.append(result)
        except Exception as e:
            logger.error(f"回测 {stock_code} 时出错: {e}", exc_info=True)
            results.append({
                'stock_code': stock_code,
                'error': str(e)
            })
    
    # 生成综合报告
    logger.info("")
    logger.info("")
    for line in log_section("综合回测报告"):
        logger.info(line)
    
    if results:
        successful_results = [r for r in results if 'error' not in r]
        
        if successful_results:
            total_profit = sum(r['profit'] for r in successful_results)
            total_return = sum(r['return_pct'] for r in successful_results)
            avg_return = total_return / len(successful_results)
            total_trades = sum(r['num_trades'] for r in successful_results)
            total_initial_cash = sum(r['initial_cash'] for r in successful_results)
            total_final_equity = sum(r['final_equity'] for r in successful_results)
            
            logger.info(f"成功回测股票数: {len(successful_results)}/{len(stock_codes)}")
            logger.info(f"总初始资金: {total_initial_cash:,.2f} 元")
            logger.info(f"总最终权益: {total_final_equity:,.2f} 元")
            logger.info(f"总盈亏: {total_profit:+,.2f} 元")
            logger.info(f"平均收益率: {avg_return:+.2f}%")
            logger.info(f"总交易次数: {total_trades}")
            logger.info("")
            
            # 详细结果表格
            logger.info("详细结果:")
            logger.info(f"{'股票代码':<15} {'初始资金':>15} {'最终权益':>15} {'盈亏':>15} {'收益率':>10} {'交易次数':>8}")
            logger.info("-" * 90)
            
            for r in successful_results:
                logger.info(
                    f"{r['stock_code']:<15} "
                    f"{r['initial_cash']:>15,.2f} "
                    f"{r['final_equity']:>15,.2f} "
                    f"{r['profit']:>+15,.2f} "
                    f"{r['return_pct']:>+10.2f}% "
                    f"{r['num_trades']:>8}"
                )
            
            # 失败的结果
            failed_results = [r for r in results if 'error' in r]
            if failed_results:
                logger.info("")
                logger.info("失败的回测:")
                for r in failed_results:
                    logger.error(f"{r['stock_code']}: {r['error']}")
            
            # 生成综合报告文件
            logger.info("")
            logger.info("生成综合报告文件...")
            
            # 从结果中提取实际的日期范围
            actual_start_date = start_date or "N/A"
            actual_end_date = end_date or "N/A"
            if successful_results and 'data_range' in successful_results[0]:
                # 尝试从第一个结果的数据范围中提取日期
                data_range = successful_results[0]['data_range']
                if ' → ' in data_range:
                    parts = data_range.split(' → ')
                    if not start_date and len(parts) > 0:
                        actual_start_date = parts[0].strip()
                    if not end_date and len(parts) > 1:
                        actual_end_date = parts[1].strip()
            
            # 获取实验文件夹名称作为输出目录名称
            experiment_dir = Path(__file__).parent.name
            from trader.config import PROJECT_ROOT
            output_dir = PROJECT_ROOT / 'output' / 'backtest' / experiment_dir
            # 不传入 title，直接使用 output_dir，避免创建子目录
            report = BacktestReport(output_dir=output_dir, title=None)
            
            strategy_params = {
                "max_position_weight": f"{max_position_weight*100:.0f}%",
                "min_score_threshold": min_score_threshold,
                "max_total_weight": f"{max_total_weight*100:.0f}%",
                "use_close_only": use_close_only,
                "seq_len": seq_len,
                "use_square_weight": use_square_weight
            }
            
            report_file = report.generate_multi_stock_report(
                results=results,
                strategy_name="Predictor Agent Strategy",
                start_date=actual_start_date,
                end_date=actual_end_date,
                initial_cash_per_stock=initial_cash,
                strategy_params=strategy_params
            )
            logger.info(f"综合报告已保存: {report_file}")
        else:
            logger.error("所有回测都失败了")
    else:
        logger.error("没有回测结果")
    
    logger.info(log_separator())


if __name__ == "__main__":
    # 从 model_config.toml 读取 test stocks（10只股票）
    test_stocks = get_test_stocks()
    
    # 运行所有股票的baseline回测（每只股票单独测试）
    # 使用全量数据，不进行训练/测试分割，和实验1.1、1.2一样
    # 
    # 参数优化说明：
    # - max_position_weight=0.5 (50%): 提高单股最大仓位，充分利用预测信号
    # - min_score_threshold=0.0: 降低阈值，允许更多交易机会
    #   注意：PredictorAgent的score使用tanh映射，缩放因子已调整为50
    #   例如：预测收益率0.1% -> score ≈ 0.05，预测收益率0.2% -> score ≈ 0.10
    # - use_square_weight=False: 改用线性映射，避免平方后weight过小
    #   由于预测收益率很小，平方映射会让weight变得极小（0.000001级别）
    #   线性映射：score=0.05 -> weight=2.5%, score=0.10 -> weight=5%
    # - max_total_weight=1.0 (100%): 允许满仓操作
    # - min_trade_amount=10.0: 降低交易阈值到10元，允许小金额交易
    run_all_stocks_baseline(
        stock_codes=test_stocks,
        initial_cash=1000000.0,
        max_position_weight=0.5,  # 单个股票最大50%（提高仓位，充分利用预测信号）
        min_score_threshold=0.0,  # 最小score阈值0.0，允许所有正分信号
        max_total_weight=1.0,  # 总仓位上限100%
        use_close_only=True,  # 只使用close_price
        seq_len=21,  # 序列长度21天
        use_square_weight=False,  # 改用线性映射，避免平方后weight过小
        start_date="2023-01-01",
        end_date="2023-12-31"
    )

