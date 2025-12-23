"""
多资产交易策略示例
使用 MultiAssetTradingAgent 对指定股票池中的股票分别使用独立的 LogisticAgent 获取信号
然后使用 multiagent_weight_normalized 进行权重归一化
"""
import sys
from pathlib import Path
from typing import List, Optional

# 添加项目根目录到 Python 路径
project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from trader.agent.MultiAssetTradingAgent import MultiAssetTradingAgent
from trader.backtest.account import Account
from trader.backtest.market import Market
from trader.backtest.engine import BacktestEngine
from trader.logger import get_logger, log_separator, log_section

logger = get_logger(__name__)


def multi_asset_strategy(
    stock_codes: Optional[List[str]] = None,
    initial_cash: float = 1000000.0,
    max_position_weight: float = 0.2,  # 单个股票最大20%
    min_score_threshold: float = 0.0,
    max_total_weight: float = 1.0,  # 总仓位上限100%
    train_window_days: int = 252,
    prediction_horizon: int = 5,
    ret_threshold: float = 0.0,
    retrain_frequency: int = 20,
    train_test_split_ratio: float = 0.7,
    start_date: str = None,
    end_date: str = None
):
    """
    多资产交易策略回测
    
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
    """
    for line in log_section("多资产交易策略回测"):
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
    
    # 生成报告标题
    report_title = (
        f"MultiAsset_Logistic_Strategy_"
        f"{len(valid_stock_codes)}stocks_maxPos{max_position_weight*100:.0f}"
    )
    engine = BacktestEngine(
        account, market,
        report_title=report_title,
        train_test_split_ratio=train_test_split_ratio
    )
    
    # 创建多资产交易代理
    agent = MultiAssetTradingAgent(
        stock_codes=valid_stock_codes,
        name="MultiAsset_Logistic",
        max_position_weight=max_position_weight,
        min_score_threshold=min_score_threshold,
        max_total_weight=max_total_weight,
        train_window_days=train_window_days,
        prediction_horizon=prediction_horizon,
        ret_threshold=ret_threshold,
        retrain_frequency=retrain_frequency,
        train_test_split_ratio=train_test_split_ratio
    )
    
    # 注册交易日回调：执行策略
    def on_trading_day(eng: BacktestEngine, date: str):
        """每个交易日的回调"""
        # 更新 agent 状态
        agent.on_date(eng, date)
        
        # 获取所有股票的归一化权重
        weights = agent.get_all_weights(eng)
        
        # 获取账户权益
        market_prices = eng.get_market_prices(valid_stock_codes)
        account_equity = account.equity(market_prices)
        
        # 对每支股票执行交易
        for stock_code in valid_stock_codes:
            weight = weights.get(stock_code, 0.0)
            
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
                        shares_to_sell = min(shares_to_sell, position['shares'])
                        if shares_to_sell > 0:
                            eng.sell(stock_code, shares=shares_to_sell)
    
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


if __name__ == "__main__":
    # 执行多资产交易策略
    # 默认使用指定的5支股票：AAPL.O, AMZN.O, ASML.O, META.O, MRNA.O
    multi_asset_strategy(
        stock_codes=["AAPL.O", "AMZN.O", "ASML.O", "META.O", "MRNA.O"],
        initial_cash=1000000.0,
        max_position_weight=0.2,  # 单个股票最大20%
        min_score_threshold=0.0,
        max_total_weight=1.0,  # 总仓位上限100%
        train_window_days=252,
        prediction_horizon=5,
        ret_threshold=0.0,
        retrain_frequency=20,
        train_test_split_ratio=0.7,
        start_date=None,
        end_date=None
    )

