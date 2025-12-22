"""
DCA (Dollar Cost Averaging) 定投策略
对 AAPL.O 进行定投，初始资金 10,000 元
使用回测引擎，支持行为队列和市场日期循环
"""
import sys
from pathlib import Path

# 添加项目根目录到 Python 路径
project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from datetime import datetime
from trader.backtest.account import Account
from trader.backtest.market import Market
from trader.backtest.engine import BacktestEngine
from trader.logger import get_logger

logger = get_logger(__name__)


def dca_strategy(
    stock_code: str = "AAPL.O",
    initial_cash: float = 10000.0,
    monthly_investment: float = 1000.0,
    start_date: str = None,
    end_date: str = None
):
    """
    定投策略回测
    
    Args:
        stock_code: 股票代码
        initial_cash: 初始资金（元）
        monthly_investment: 每月定投金额（元）
        start_date: 开始日期（格式: YYYY-MM-DD），如果为 None 则从最早可用日期开始
        end_date: 结束日期（格式: YYYY-MM-DD），如果为 None 则到最新日期
    """
    logger.info("=" * 60)
    logger.info("DCA 定投策略回测")
    logger.info("=" * 60)
    logger.info(f"股票代码: {stock_code}")
    logger.info(f"初始资金: {initial_cash:,.2f} 元")
    logger.info(f"每月定投: {monthly_investment:,.2f} 元")
    
    # 初始化市场、账户和回测引擎
    # 注意：如果价格数据单位不对（如价格过高），可以调整 price_adjustment
    # 例如：如果价格需要除以100，设置 price_adjustment=0.01
    # 如果价格需要除以1000，设置 price_adjustment=0.001
    market = Market(price_adjustment=0.01)  # 假设价格单位是分，需要除以100
    account = Account(initial_cash=initial_cash)
    engine = BacktestEngine(account, market)
    
    # 获取可用日期（用于显示数据范围）
    available_dates = market.get_available_dates(stock_code)
    if not available_dates:
        logger.error(f"未找到股票 {stock_code} 的数据")
        return
    
    logger.info(f"数据范围: {available_dates[0]} 至 {available_dates[-1]}")
    
    # 定投策略：每月第一个交易日买入
    last_month = None
    investment_count = [0]  # 使用列表以便在回调中修改
    
    def on_trading_day(eng: BacktestEngine, date: str):
        """每个交易日的回调函数"""
        nonlocal last_month  # 必须在函数开头声明
        
        # 解析日期，获取年月
        date_obj = datetime.strptime(date, "%Y-%m-%d")
        current_month = (date_obj.year, date_obj.month)
        
        # 如果是新的月份，提交定投买入订单
        if current_month != last_month:
            # 获取当日价格
            price = eng.get_current_price(stock_code)
            
            if price is None:
                logger.warning(f"日期 {date} 无法获取价格，跳过")
                return
            
            # 提交买入订单（按金额买入）
            eng.buy(stock_code, amount=monthly_investment)
            investment_count[0] += 1
            
            logger.info(
                f"[{date}] 提交定投买入订单: {monthly_investment:.2f} 元 @ {price:.2f}"
            )
            
            # 更新月份
            last_month = current_month
    
    # 注册交易日回调
    engine.on_date(on_trading_day)
    
    # 运行回测
    engine.run(stock_code, start_date=start_date, end_date=end_date)
    
    # 计算最终结果
    final_price = market.get_price(stock_code, end_date if end_date else available_dates[-1])
    if final_price is None:
        final_price = market.get_price(stock_code)  # 使用最新价格
    
    if final_price is None:
        logger.error("无法获取最终价格")
        return
    
    market_prices = {stock_code: final_price}
    equity = account.equity(market_prices)
    profit = account.get_total_profit(market_prices)
    return_pct = account.get_total_return(market_prices)
    
    # 计算实际总投入（初始资金 - 剩余现金）
    total_invested = initial_cash - account.cash
    
    # 输出结果
    logger.info("")
    logger.info("=" * 60)
    logger.info("回测结果")
    logger.info("=" * 60)
    logger.info(f"定投次数: {investment_count[0]}")
    logger.info(f"初始资金: {initial_cash:,.2f} 元")
    logger.info(f"实际投入: {total_invested:,.2f} 元")
    logger.info(f"当前现金: {account.cash:,.2f} 元")
    
    position = account.get_position(stock_code)
    if position:
        position_value = position["shares"] * final_price
        position_profit = (final_price - position["average_price"]) * position["shares"]
        logger.info(f"持仓股数: {position['shares']} 股")
        logger.info(f"平均成本: {position['average_price']:.2f} 元")
        logger.info(f"当前价格: {final_price:.2f} 元")
        logger.info(f"持仓市值: {position_value:,.2f} 元")
        logger.info(f"持仓盈亏: {position_profit:+,.2f} 元")
    
    logger.info(f"总权益: {equity:,.2f} 元")
    logger.info(f"总盈亏: {profit:+,.2f} 元")
    logger.info(f"总收益率: {return_pct:+.2f}%")
    logger.info("=" * 60)
    
    # 输出详细账户摘要
    logger.info("")
    logger.info(account.summary(market_prices))


if __name__ == "__main__":
    # 执行定投策略
    dca_strategy(
        stock_code="AAPL.O",
        initial_cash=10000.0,
        monthly_investment=1000.0,
        start_date=None,  # 从最早日期开始
        end_date=None     # 到最新日期
    )
