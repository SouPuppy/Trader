"""
Logistic Regression 逻辑回归策略回测示例（增加投资比例）
使用 LogisticAgent 实现基于逻辑回归的预测策略，增加投资比例以更明显地看出效果

参数设置：
- max_position_weight=0.5 (50%) - 单个股票最大仓位（增加投资力度，方便风险控制）
- max_total_weight=1.0 (100%) - 总仓位上限

对 trader/config.toml 中的所有 test 股票分别进行单股票回测，生成综合报告
"""
import sys
from pathlib import Path
from typing import List, Optional, Dict

# 添加项目根目录到 Python 路径
project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from trader.agent.agent_logistic import LogisticAgent
from trader.backtest.account import Account
from trader.backtest.market import Market
from trader.backtest.engine import BacktestEngine
from trader.backtest.report import BacktestReport
from trader.config_loader import get_test_stocks
from trader.logger import get_logger, log_separator, log_section

logger = get_logger(__name__)


def logistic_strategy_single(
    stock_code: str = "AAPL.O",
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
    start_date: str = None,
    end_date: str = None
) -> Optional[Dict]:
    """
    逻辑回归策略回测（使用 LogisticAgent，增加投资比例）- 单股票版本
    
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
        start_date: 开始日期
        end_date: 结束日期
    
    Returns:
        回测结果字典，包含 stock_code, initial_cash, final_equity, profit, return_pct, num_trades 等信息
    """
    for line in log_section("逻辑回归策略回测（增加投资比例）"):
        logger.info(line)
    logger.info(f"股票代码: {stock_code}")
    logger.info(f"初始资金: {initial_cash:,.2f} 元")
    logger.info(f"训练窗口: {train_window_days} 天")
    logger.info(f"预测周期: {prediction_horizon} 天")
    logger.info(f"收益阈值: {ret_threshold:.2%}")
    logger.info(f"重新训练频率: 每 {retrain_frequency} 个交易日")
    logger.info(f"单股最大仓位: {max_position_weight*100:.0f}%")
    logger.info(f"总仓位上限: {max_total_weight*100:.0f}%")
    logger.info(f"训练/测试分割比例: {train_test_split_ratio*100:.0f}% / {(1-train_test_split_ratio)*100:.0f}%")
    
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
    
    # 创建逻辑回归 Agent
    agent = LogisticAgent(
        name="Logistic_Strategy",
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
    
    # 注册交易日回调：执行策略
    def on_trading_day(eng: BacktestEngine, date: str):
        """每个交易日的回调"""
        # 更新 agent 状态（更新交易日计数）
        agent.on_date(eng, date)
        
        # 获取当前股票的 score
        score = agent.score(stock_code, eng)
        
        # 计算权重
        weight = agent.weight(stock_code, score, eng)
        
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
        
        # 执行交易（最小交易金额阈值：100元）
        if abs(diff_value) > 100:
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
    final_price = market.get_price(stock_code, end_date if end_date else available_dates[-1])
    if final_price is None:
        final_price = market.get_price(stock_code)
    
    if final_price is None:
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
    logger.info(f"模型训练次数: {agent.train_count}")
    logger.info(log_separator())
    
    # 输出详细账户摘要
    logger.info("")
    logger.info(account.summary(market_prices))
    
    # 返回结果字典，用于生成综合报告
    actual_start_date = start_date or available_dates[0]
    actual_end_date = end_date or available_dates[-1]
    
    return {
        'stock_code': stock_code,
        'initial_cash': initial_cash,
        'final_equity': equity,
        'profit': profit,
        'return_pct': return_pct,
        'num_trades': len(account.trades),
        'train_count': agent.train_count,
        'data_range': f"{available_dates[0]} → {available_dates[-1]}",
        # 保存账户和引擎信息，用于生成图表和详细指标
        'account': account,
        'engine': engine,
        'market': market,
        'start_date': actual_start_date,
        'end_date': actual_end_date
    }


def logistic_strategy_all_stocks(
    stock_codes: Optional[List[str]] = None,
    initial_cash: float = 1000000.0,
    feature_names: list = None,
    train_window_days: int = 252,
    prediction_horizon: int = 5,
    ret_threshold: float = 0.0,
    retrain_frequency: int = 20,
    max_position_weight: float = 0.5,
    min_score_threshold: float = 0.0,
    max_total_weight: float = 1.0,
    train_test_split_ratio: float = 0.7,
    start_date: str = None,
    end_date: str = None
):
    """
    对所有股票分别进行逻辑回归策略回测，生成综合报告
    
    Args:
        stock_codes: 股票代码列表，如果为 None 则从 trader/config.toml 读取
        其他参数同 logistic_strategy_single
    """
    # 如果没有指定股票代码，从配置文件读取
    if stock_codes is None:
        stock_codes = get_test_stocks()
    
    for line in log_section("逻辑回归策略回测（增加投资比例）- 多股票测试"):
        logger.info(line)
    logger.info(f"股票数量: {len(stock_codes)}")
    logger.info(f"股票列表: {stock_codes}")
    logger.info(f"初始资金: {initial_cash:,.2f} 元/股票")
    logger.info("注意：每只股票使用独立的账户，分别测试")
    logger.info("")
    
    # 运行所有股票的回测
    results = []
    for i, stock_code in enumerate(stock_codes, 1):
        logger.info("")
        logger.info(f"========== [{i}/{len(stock_codes)}] {stock_code} ==========")
        
        try:
            result = logistic_strategy_single(
                stock_code=stock_code,
                initial_cash=initial_cash,
                feature_names=feature_names,
                train_window_days=train_window_days,
                prediction_horizon=prediction_horizon,
                ret_threshold=ret_threshold,
                retrain_frequency=retrain_frequency,
                max_position_weight=max_position_weight,
                min_score_threshold=min_score_threshold,
                max_total_weight=max_total_weight,
                train_test_split_ratio=train_test_split_ratio,
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
            total_initial_cash = sum(r['initial_cash'] for r in successful_results)
            total_final_equity = sum(r['final_equity'] for r in successful_results)
            total_trades = sum(r['num_trades'] for r in successful_results)
            
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
                "train_window_days": train_window_days,
                "prediction_horizon": prediction_horizon,
                "ret_threshold": f"{ret_threshold:.2%}",
                "retrain_frequency": f"每 {retrain_frequency} 个交易日",
                "max_position_weight": f"{max_position_weight*100:.0f}%",
                "min_score_threshold": min_score_threshold,
                "max_total_weight": f"{max_total_weight*100:.0f}%",
                "train_test_split_ratio": f"{train_test_split_ratio*100:.0f}%"
            }
            
            report_file = report.generate_multi_stock_report(
                results=results,
                strategy_name="Logistic Regression Strategy (Higher Position)",
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
    # 对所有 test 股票分别进行逻辑回归策略回测，生成综合报告
    logistic_strategy_all_stocks(
        stock_codes=None,  # 从 trader/config.toml 读取
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
        start_date=None,  # 从最早日期开始
        end_date=None     # 到最新日期
    )

