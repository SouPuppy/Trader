"""
实验 5.3：单股票交易系统（不带反思层）- MRNA.O

Layer A: Signal Layer（信号层）
Layer B: Allocation & RiskControl Layer（分配与风控层）

在单股票 MRNA.O 的基础上，参数θ在整个回测期间保持不变，不做任何反思调整。

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


def single_stock_strategy_without_reflection(
    stock_code: str = "MRNA.O",
    initial_cash: float = 1000000.0,
    initial_theta: Theta = None,
    start_date: str = None,
    end_date: str = None,
    train_test_split_ratio: float = 0.7
):
    """
    单股票交易策略（不带反思层，参数θ保持不变）
    
    Args:
        stock_code: 股票代码
        initial_cash: 初始资金
        initial_theta: 初始参数θ（如果为None则使用默认值）
        start_date: 开始日期
        end_date: 结束日期
        train_test_split_ratio: 训练/测试分割比例
    """
    if initial_theta is None:
        initial_theta = Theta(
            gross_exposure=0.85,  # 提高总仓位到85%，充分利用资金
            max_w=0.20,  # 单票上限20%，单股票场景下实际就是总仓位
            turnover_cap=0.25,  # 适度提高换手率上限，允许灵活调整
            risk_mode="neutral",
            enter_th=0.02,  # 降低进场阈值，让更多股票有机会
            exit_th=-0.10  # 放宽出场阈值，避免过早止损
        )
    
    for line in log_section("单股票交易系统（不带反思层）"):
        logger.info(line)
    logger.info(f"股票代码: {stock_code}")
    logger.info(f"初始资金: {initial_cash:,.2f} 元")
    logger.info(f"初始参数θ: {initial_theta}")
    logger.info("注意：参数θ在整个回测期间保持不变，不做任何反思调整")
    logger.info("")
    
    # 初始化市场、账户和回测引擎
    market = Market(price_adjustment=0.01)
    account = Account(initial_cash=initial_cash)
    
    # 设置输出目录
    experiment_dir = Path(__file__).parent.name
    from trader.config import PROJECT_ROOT
    output_dir = PROJECT_ROOT / 'output' / 'backtest' / experiment_dir
    
    # 创建ReflectionEngine（支持参数追踪，但参数不会变化）
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
        
        # 获取账户权益
        current_price = eng.get_current_price(stock_code)
        if current_price:
            market_prices = {stock_code: current_price}
        else:
            market_prices = {}
        account_equity = account.equity(market_prices)
        
        # 获取目标权重（使用固定参数）
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
        
        # 如果变化很小（小于账户权益的0.5%），不需要交易
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
    logger.info(f"参数变化次数: {len(engine.theta_history)} (应该为1，因为参数保持不变)")
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
        strategy_name="单股票交易系统（不带反思层）",
        is_single_stock=True
    )
    
    logger.info(f"报告已保存: {report_file}")


if __name__ == "__main__":
    # 执行单股票交易策略（不带反思层）
    single_stock_strategy_without_reflection(
        stock_code="MRNA.O",
        initial_cash=1000000.0,
        initial_theta=Theta(
            gross_exposure=0.85,  # 提高总仓位到85%，充分利用资金
            max_w=0.20,  # 单票上限20%，单股票场景下实际就是总仓位
            turnover_cap=0.25,  # 适度提高换手率上限，允许灵活调整
            risk_mode="neutral",
            enter_th=0.02,  # 降低进场阈值，让更多股票有机会
            exit_th=-0.10  # 放宽出场阈值，避免过早止损
        ),
        start_date=None,
        end_date=None,
        train_test_split_ratio=0.7
    )

