"""
多资产交易策略示例（使用 TurtleAgent）
使用 MultiAssetTurtleAgent 对指定股票池中的股票分别使用独立的 TurtleAgent 获取信号
然后使用 multiagent_weight_normalized 进行权重归一化
"""
import sys
from pathlib import Path
from typing import List, Optional

# 添加项目根目录到 Python 路径
project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from trader.agent.MultiAssetTurtleAgent import MultiAssetTurtleAgent
from trader.backtest.account import Account
from trader.backtest.market import Market
from trader.backtest.engine import BacktestEngine
from trader.logger import get_logger, log_separator, log_section

logger = get_logger(__name__)


def multi_asset_strategy(
    stock_codes: Optional[List[str]] = None,
    initial_cash: float = 1000000.0,
    max_position_weight: float = 0.4,  # 单个股票最大40%
    min_score_threshold: float = 0.0,  # TurtleAgent 的 score 范围是 [-1, 1]
    max_total_weight: float = 1.0,  # 总仓位上限100%
    # TurtleAgent 参数
    entry_period: int = 20,  # 突破周期（N日最高/最低）
    exit_period: int = 10,   # 退出周期
    atr_period: int = 20,    # ATR计算周期
    risk_per_trade: float = 0.02,  # 每次交易风险（账户资金的百分比）
    stop_loss_atr: float = 2.0,   # 止损距离（ATR倍数）
    max_positions: int = 4,       # 最大加仓次数
    add_position_atr: float = 0.5,  # 加仓距离（ATR倍数）
    start_date: str = None,
    end_date: str = None,
    # 交易频率控制参数
    min_trade_amount: float = 5000.0,  # 最小交易金额阈值
    min_weight_change: float = 0.05,  # 最小权重变化阈值（5%），避免微小调整
):
    """
    多资产交易策略回测（使用 TurtleAgent）
    
    Args:
        stock_codes: 股票代码列表，如果为 None 则使用默认股票池
        initial_cash: 初始资金
        max_position_weight: 单个股票最大配置比例
        min_score_threshold: 最小 score 阈值
        max_total_weight: 总配置比例上限
        entry_period: TurtleAgent 突破周期
        exit_period: TurtleAgent 退出周期
        atr_period: TurtleAgent ATR计算周期
        risk_per_trade: TurtleAgent 每次交易风险
        stop_loss_atr: TurtleAgent 止损距离（ATR倍数）
        max_positions: TurtleAgent 最大加仓次数
        add_position_atr: TurtleAgent 加仓距离（ATR倍数）
        start_date: 开始日期
        end_date: 结束日期
        min_trade_amount: 最小交易金额阈值
        min_weight_change: 最小权重变化阈值
    """
    for line in log_section("多资产交易策略回测（Turtle）"):
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
    logger.info(f"Turtle 参数: entry={entry_period}, exit={exit_period}, ATR={atr_period}, risk={risk_per_trade*100:.1f}%")
    
    # 生成报告标题
    report_title = (
        f"MultiAsset_Turtle_Strategy_"
        f"{len(valid_stock_codes)}stocks_maxPos{max_position_weight*100:.0f}"
    )
    engine = BacktestEngine(
        account, market,
        report_title=report_title,
        train_test_split_ratio=0.0  # Turtle 策略不需要训练/测试分割
    )
    
    # 创建多资产交易代理（使用 TurtleAgent）
    agent = MultiAssetTurtleAgent(
        stock_codes=valid_stock_codes,
        name="MultiAsset_Turtle",
        max_position_weight=max_position_weight,
        min_score_threshold=min_score_threshold,
        max_total_weight=max_total_weight,
        entry_period=entry_period,
        exit_period=exit_period,
        atr_period=atr_period,
        risk_per_trade=risk_per_trade,
        stop_loss_atr=stop_loss_atr,
        max_positions=max_positions,
        add_position_atr=add_position_atr,
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
        
        # 获取所有股票的归一化权重
        weights = agent.get_all_weights(eng)
        
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


if __name__ == "__main__":
    # 执行多资产交易策略（使用 TurtleAgent）
    # 默认使用指定的5支股票：AAPL.O, AMZN.O, ASML.O, META.O, MRNA.O
    multi_asset_strategy(
        stock_codes=["AAPL.O", "AMZN.O", "ASML.O", "META.O", "MRNA.O"],
        initial_cash=1000000.0,
        max_position_weight=0.4,  # 单个股票最大40%
        min_score_threshold=0.0,  # TurtleAgent 的 score 范围是 [-1, 1]
        max_total_weight=1.0,  # 总仓位上限100%
        # TurtleAgent 参数
        entry_period=20,  # 突破周期
        exit_period=10,   # 退出周期
        atr_period=20,    # ATR计算周期
        risk_per_trade=0.02,  # 每次交易风险2%
        stop_loss_atr=2.0,   # 止损距离（2倍ATR）
        max_positions=4,       # 最大加仓次数
        add_position_atr=0.5,  # 加仓距离（0.5倍ATR）
        start_date=None,
        end_date=None,
        # 优化参数：减少频繁交易
        min_trade_amount=5000.0,  # 最小交易金额5000元
        min_weight_change=0.05,  # 权重变化至少5%才交易
    )

