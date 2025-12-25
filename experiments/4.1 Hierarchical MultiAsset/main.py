"""
实验 4.1：层级式多资产交易系统（无反思层）- 优化版

Layer A: Signal Layer（信号层）
Layer B: Allocation & RiskControl Layer（分配与风控层）
Layer C: 无反思层（参数θ固定）

使用ReflectionEngine追踪参数变化，但参数不会自动调整

优化内容：
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
from pathlib import Path

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


def hierarchical_multiasset_strategy(
    stock_codes: list = None,
    initial_cash: float = 1000000.0,
    initial_theta: Theta = None,
    start_date: str = None,
    end_date: str = None,
    train_test_split_ratio: float = 0.7
):
    """
    层级式多资产交易策略（无反思层）
    
    Args:
        stock_codes: 股票代码列表
        initial_cash: 初始资金
        initial_theta: 初始参数θ（如果为None则使用默认值）
        start_date: 开始日期
        end_date: 结束日期
        train_test_split_ratio: 训练/测试分割比例
    """
    if stock_codes is None:
        stock_codes = ["AAPL.O", "MSFT.O", "GOOGL.O", "AMZN.O", "NVDA.O"]
    
    if initial_theta is None:
        initial_theta = Theta(
            gross_exposure=0.7,  # 进一步降低总仓位，更保守
            max_w=0.15,  # 降低单票上限，分散风险
            turnover_cap=0.20,  # 进一步降低换手率，减少交易成本
            risk_mode="neutral",
            enter_th=0.02,  # 降低进场阈值，让更多股票有机会
            exit_th=-0.10  # 放宽出场阈值，避免过早止损
        )
    
    for line in log_section("层级式多资产交易系统（无反思层）"):
        logger.info(line)
    logger.info(f"股票代码: {', '.join(stock_codes)}")
    logger.info(f"初始资金: {initial_cash:,.2f} 元")
    logger.info(f"初始参数θ: {initial_theta}")
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
        train_test_split_ratio=train_test_split_ratio
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
        w_T=0.5,  # Trend权重（提高，趋势最重要）
        w_V=0.15,  # Volatility权重（降低）
        w_F=0.20,  # Fundamental权重（保持）
        w_N=0.15,  # News权重（降低，因为可能不稳定）
        theta=initial_theta
    )
    
    # 创建Layer B: Constrained Allocator
    allocator = ConstrainedAllocator(
        signal_layer=signal_layer,
        theta=initial_theta,
        name="ConstrainedAllocator"
    )
    
    # 注册交易日回调
    def on_trading_day(eng: ReflectionEngine, date: str):
        """每个交易日的回调"""
        # 记录每日参数（即使没有变化）
        eng.record_daily_theta(date)
        
        # 获取所有股票的目标权重
        target_weights = allocator.get_weights(stock_codes, eng)
        
        # 获取账户权益
        market_prices = eng.get_market_prices(stock_codes)
        account_equity = account.equity(market_prices)
        
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
            # 这样可以减少不必要的交易，降低交易成本
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
        strategy_name="层级式多资产交易系统（无反思层）"
    )
    
    logger.info(f"报告已保存: {report_file}")


if __name__ == "__main__":
    # 执行层级式多资产交易策略（无反思层）
    hierarchical_multiasset_strategy(
        stock_codes=["AAPL.O", "MSFT.O", "GOOGL.O", "AMZN.O", "NVDA.O"],
        initial_cash=1000000.0,
        initial_theta=Theta(
            gross_exposure=0.7,  # 进一步降低总仓位，更保守
            max_w=0.15,  # 降低单票上限，分散风险
            turnover_cap=0.20,  # 进一步降低换手率，减少交易成本
            risk_mode="neutral",
            enter_th=0.02,  # 降低进场阈值，让更多股票有机会
            exit_th=-0.10  # 放宽出场阈值，避免过早止损
        ),
        start_date=None,
        end_date=None,
        train_test_split_ratio=0.7
    )

