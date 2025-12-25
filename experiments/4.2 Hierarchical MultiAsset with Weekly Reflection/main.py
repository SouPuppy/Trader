"""
实验 4.2：层级式多资产交易系统（带每周反思层）

Layer A: Signal Layer（信号层）
Layer B: Allocation & RiskControl Layer（分配与风控层）
Layer C: 每周反思层（参数θ每周调整）

在 4.1 的基础上，每周进行一次 naive reflection：
1. 评估过去一周的表现（周收益率）
2. 根据表现调整参数：
   - 如果周收益率 > 1%（表现好）：稍微激进（增加仓位、降低阈值）
   - 如果周收益率 < -1%（表现差）：更保守（降低仓位、提高阈值）
   - 如果周收益率在 -1% 到 1% 之间：保持参数不变或微调

优化内容（继承自 4.1）：
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


def naive_reflection(current_theta: Theta, weekly_return: float, reflection_id: int) -> Theta:
    """
    保守的反思策略：只在表现明显好或差时才调整参数，避免过度反应
    
    Args:
        current_theta: 当前参数θ
        weekly_return: 周收益率（小数，例如 0.01 表示 1%）
        reflection_id: 反思轮次ID
        
    Returns:
        Theta: 调整后的参数θ
    """
    new_theta = current_theta.copy()
    new_theta.reflection_id = reflection_id
    
    # 更保守的调整策略：
    # 1. 大幅提高阈值：只在周收益率 > 1.5% 或 < -1.5% 时才调整，减少频繁调整
    positive_threshold = 0.015  # 1.5%（大幅提高，减少调整频率）
    negative_threshold = -0.015  # -1.5%
    
    # 2. 降低调整幅度：从3%降到2%，使调整更平滑
    adjustment_factor = 0.02  # 2% 的调整幅度（更保守）
    
    # 3. 移除微调逻辑：只在明显好/差时才调整，避免噪音干扰
    
    if weekly_return > positive_threshold:  # 周收益率 > 1.5%，表现明显好
        # 稍微激进：增加仓位、降低阈值
        logger.info(f"[Reflection] 周收益率 {weekly_return*100:.2f}% > {positive_threshold*100:.2f}%，表现良好，适度激进调整")
        
        # 增加总仓位（但不超过上限，且更保守的增长）
        new_theta.gross_exposure = min(0.85, current_theta.gross_exposure * (1 + adjustment_factor))
        
        # 增加单票上限（但不超过上限）
        new_theta.max_w = min(0.20, current_theta.max_w * (1 + adjustment_factor))
        
        # 适度增加换手上限（允许更多交易，但更保守）
        new_theta.turnover_cap = min(0.30, current_theta.turnover_cap * (1 + adjustment_factor * 0.6))
        
        # 风险模式：保持中性，不轻易切换
        # 只在从 risk_off 切换到 neutral，不直接跳到 risk_on
        if current_theta.risk_mode == "risk_off":
            new_theta.risk_mode = "neutral"
        # neutral 和 risk_on 保持不变
        
        # 降低进场阈值（更容易进场，但保持合理下限）
        new_theta.enter_th = max(0.01, current_theta.enter_th * (1 - adjustment_factor * 0.8))
        
        # 适度放宽出场阈值（更不容易止损）
        new_theta.exit_th = min(-0.08, current_theta.exit_th * (1 - adjustment_factor * 0.6))
        
    elif weekly_return < negative_threshold:  # 周收益率 < -1.5%，表现明显差
        # 更保守：降低仓位、提高阈值
        logger.info(f"[Reflection] 周收益率 {weekly_return*100:.2f}% < {negative_threshold*100:.2f}%，表现不佳，更保守调整")
        
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
        # risk_off 保持不变
        
        # 提高进场阈值（更难进场）
        new_theta.enter_th = min(0.04, current_theta.enter_th * (1 + adjustment_factor * 0.8))
        
        # 适度收紧出场阈值（更容易止损，但不过度）
        new_theta.exit_th = max(-0.12, current_theta.exit_th * (1 - adjustment_factor * 0.6))
        
    else:
        # 在 ±1.5% 范围内，保持参数不变（大幅减少调整频率）
        logger.info(f"[Reflection] 周收益率 {weekly_return*100:.2f}% 在正常波动范围（±{positive_threshold*100:.2f}%），保持参数不变")
    
    logger.info(f"[Reflection] 参数调整: {current_theta} -> {new_theta}")
    return new_theta


def hierarchical_multiasset_strategy_with_reflection(
    stock_codes: list = None,
    initial_cash: float = 1000000.0,
    initial_theta: Theta = None,
    start_date: str = None,
    end_date: str = None,
    train_test_split_ratio: float = 0.7
):
    """
    层级式多资产交易策略（带每周反思层）
    
    Args:
        stock_codes: 股票代码列表
        initial_cash: 初始资金
        initial_theta: 初始参数θ（如果为None则使用默认值）
        start_date: 开始日期
        end_date: 结束日期
        train_test_split_ratio: 训练/测试分割比例
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
    
    for line in log_section("层级式多资产交易系统（带每周反思层）"):
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
    
    # 跟踪每周权益（用于计算周收益率）
    weekly_equity_history: Dict[str, float] = {}  # {date: equity}
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
                # 找到上周五的权益
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
                    # 有交易，使用账户权益
                    weekly_return = (account_equity - account.initial_cash) / account.initial_cash
                    logger.info(f"[Reflection] 第一次反思，使用账户权益作为基准")
                    logger.info(f"[Reflection] 初始资金: {account.initial_cash:,.2f}, 当前权益: {account_equity:,.2f}")
                    logger.info(f"[Reflection] 周收益率: {weekly_return*100:.2f}%")
                else:
                    # 没有交易，使用市场表现
                    use_market_return = True
            
            # 如果没有交易或权益没有变化，使用市场表现（组合股票的加权收益率）
            if use_market_return:
                # 计算组合股票的周收益率
                market_return = 0.0
                valid_stocks = 0
                
                if last_reflection_date:
                    # 计算每只股票从上周五到本周五的收益率
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
                        market_return = market_return / valid_stocks  # 平均收益率
                        weekly_return = market_return
                        logger.info(f"[Reflection] 计算周收益率（市场表现）: {last_reflection_date} -> {date}")
                        logger.info(f"[Reflection] 组合平均周收益率: {weekly_return*100:.2f}% (基于 {valid_stocks} 只股票)")
                    else:
                        logger.warning(f"[Reflection] 无法计算市场收益率，跳过本次反思")
                        return
                else:
                    # 第一次反思，使用第一周的市场表现
                    # 找到第一个交易日
                    if available_dates and len(available_dates) > 5:
                        first_date = available_dates[0]
                        try:
                            # 计算第一周的市场表现
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
            
            # 执行 naive reflection（只要成功计算了周收益率就执行）
            # weekly_return 已经被设置（无论是账户权益还是市场表现）
            reflection_count += 1
            current_theta = eng.get_current_theta()
            new_theta = naive_reflection(current_theta, weekly_return, reflection_count)
            
            # 更新参数
            eng.update_theta(new_theta, date)
            
            # 更新 allocator 和 signal_layer 的参数（使用 update_theta 方法以确保所有属性都更新）
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
        strategy_name="层级式多资产交易系统（带每周反思层）"
    )
    
    logger.info(f"报告已保存: {report_file}")


if __name__ == "__main__":
    # 执行层级式多资产交易策略（带每周反思层）
    hierarchical_multiasset_strategy_with_reflection(
        stock_codes=[
            "AAPL.O", "MSFT.O", "GOOGL.O", "AMZN.O", "NVDA.O",
            "TSLA.O", "META.O", "ASML.O", "MRNA.O", "NFLX.O",
            "AMD.O", "INTC.O", "ADBE.O", "CRM.N", "ORCL.N",
            "CSCO.O", "JPM.N", "V.N", "MA.N", "WMT.N"
        ],
        initial_cash=1000000.0,
        initial_theta=Theta(
            gross_exposure=0.85,  # 提高总仓位到85%，充分利用资金
            max_w=0.20,  # 提高单票上限到20%，允许集中配置优质股票
            turnover_cap=0.25,  # 适度提高换手率上限，允许灵活调整
            risk_mode="neutral",
            enter_th=0.02,  # 降低进场阈值，让更多股票有机会
            exit_th=-0.10  # 放宽出场阈值，避免过早止损
        ),
        start_date=None,
        end_date=None,
        train_test_split_ratio=0.7
    )

