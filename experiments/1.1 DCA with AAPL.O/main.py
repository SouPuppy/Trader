"""
DCA (Dollar Cost Averaging) backtest
Single-asset monthly investment strategy using DCAAgent
对 trader/config.toml 中的所有股票分别进行单股票回测，生成综合报告
"""

import sys
from pathlib import Path
from typing import List, Optional, Dict

# Ensure project root is on PYTHONPATH
project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

from trader.agent.agent_dca import DCAAgent
from trader.backtest.account import Account
from trader.backtest.market import Market
from trader.backtest.engine import BacktestEngine
from trader.backtest.report import BacktestReport
from trader.config_loader import get_test_stocks
from trader.logger import get_logger, log_section, log_separator

logger = get_logger(__name__)


def run_dca_backtest_single(
    stock_code: str,
    initial_cash: float = 1_000_000.0,
    monthly_investment: float = 100_000.0,
    start_date: str | None = None,
    end_date: str | None = None,
) -> Optional[Dict]:
    """Run a DCA backtest for a single asset."""

    for line in log_section("DCA Strategy Backtest"):
        logger.info(line)

    logger.info(f"Symbol: {stock_code}")
    logger.info(f"Initial cash: {initial_cash:,.2f}")
    logger.info(f"Monthly investment: {monthly_investment:,.2f}")

    market = Market(price_adjustment=0.01)
    account = Account(initial_cash=initial_cash)

    # 启用报告生成以记录每日状态，但不生成独立报告文件（通过设置 report_title=None）
    # DCA 策略不需要训练，使用所有数据（train_test_split_ratio=0.0）
    engine = BacktestEngine(account, market, enable_report=True, report_title=None, train_test_split_ratio=0.0)

    available_dates = market.get_available_dates(stock_code)
    if not available_dates:
        logger.error(f"No market data for {stock_code}")
        return None

    agent = DCAAgent(
        name="DCA",
        monthly_investment=monthly_investment,
        dca_frequency="monthly",
    )
    agent.set_dca_stocks([stock_code])

    def on_trading_day(engine: BacktestEngine, date: str):
        agent.on_date(engine, date)

    engine.on_date(on_trading_day)

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
    invested = initial_cash - account.cash

    return {
        'stock_code': stock_code,
        'initial_cash': initial_cash,
        'final_equity': equity,
        'profit': profit,
        'return_pct': return_pct,
        'invested': invested,
        'remaining_cash': account.cash,
        'dca_executions': agent.investment_count,
        'data_range': f"{available_dates[0]} → {available_dates[-1]}",
        # 保存账户和引擎信息，用于生成图表和详细指标
        'account': account,
        'engine': engine,
        'market': market,
        'start_date': start_date or available_dates[0],
        'end_date': end_date or available_dates[-1]
    }


def run_dca_backtest_all_stocks(
    stock_codes: Optional[List[str]] = None,
    initial_cash: float = 1_000_000.0,
    monthly_investment: float = 100_000.0,
    start_date: str | None = None,
    end_date: str | None = None,
):
    """
    对所有股票分别进行 DCA 回测，生成综合报告
    
    Args:
        stock_codes: 股票代码列表，如果为 None 则从 trader/config.toml 读取
        其他参数同 run_dca_backtest_single
    """
    # 如果没有指定股票代码，从配置文件读取
    if stock_codes is None:
        stock_codes = get_test_stocks()
    
    for line in log_section("DCA Strategy Backtest - 多股票单股票测试"):
        logger.info(line)
    logger.info(f"股票数量: {len(stock_codes)}")
    logger.info(f"股票列表: {stock_codes}")
    logger.info(f"初始资金: {initial_cash:,.2f} 元/股票")
    logger.info(f"每月投资: {monthly_investment:,.2f} 元")
    logger.info("注意：每只股票使用独立的账户，分别测试")
    logger.info("")
    
    # 运行所有股票的回测
    results = []
    for i, stock_code in enumerate(stock_codes, 1):
        logger.info("")
        logger.info(f"========== [{i}/{len(stock_codes)}] {stock_code} ==========")
        
        try:
            result = run_dca_backtest_single(
                stock_code=stock_code,
                initial_cash=initial_cash,
                monthly_investment=monthly_investment,
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
            total_invested = sum(r['invested'] for r in successful_results)
            total_dca_executions = sum(r['dca_executions'] for r in successful_results)
            
            logger.info(f"成功回测股票数: {len(successful_results)}/{len(stock_codes)}")
            logger.info(f"总初始资金: {total_initial_cash:,.2f} 元")
            logger.info(f"总最终权益: {total_final_equity:,.2f} 元")
            logger.info(f"总投入资金: {total_invested:,.2f} 元")
            logger.info(f"总盈亏: {total_profit:+,.2f} 元")
            logger.info(f"平均收益率: {avg_return:+.2f}%")
            logger.info(f"总 DCA 执行次数: {total_dca_executions}")
            logger.info("")
            
            # 详细结果表格
            logger.info("详细结果:")
            logger.info(f"{'股票代码':<15} {'初始资金':>15} {'最终权益':>15} {'盈亏':>15} {'收益率':>10} {'DCA次数':>8}")
            logger.info("-" * 90)
            
            for r in successful_results:
                logger.info(
                    f"{r['stock_code']:<15} "
                    f"{r['initial_cash']:>15,.2f} "
                    f"{r['final_equity']:>15,.2f} "
                    f"{r['profit']:>+15,.2f} "
                    f"{r['return_pct']:>+10.2f}% "
                    f"{r['dca_executions']:>8}"
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
                "monthly_investment": f"{monthly_investment:,.2f} 元",
                "dca_frequency": "monthly"
            }
            
            report_file = report.generate_multi_stock_report(
                results=results,
                strategy_name="DCA Strategy",
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
    # 对所有股票分别进行 DCA 回测
    run_dca_backtest_all_stocks(
        stock_codes=None,  # 从 trader/config.toml 读取
        initial_cash=1_000_000.0,
        monthly_investment=100_000.0,
        start_date="2023-01-01",
        end_date="2023-12-31",
    )
