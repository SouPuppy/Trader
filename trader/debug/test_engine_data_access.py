"""
测试engine的数据访问接口和日期保护功能
"""
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from trader.backtest.account import Account
from trader.backtest.market import Market
from trader.backtest.engine import BacktestEngine
from trader.logger import get_logger, log_separator, log_section

logger = get_logger(__name__)


def test_engine_data_access():
    """测试engine的数据访问接口"""
    for line in log_section("测试 Engine 数据访问接口和日期保护"):
        logger.info(line)
    
    # 初始化
    market = Market(price_adjustment=0.01)
    account = Account(initial_cash=10000.0)
    engine = BacktestEngine(account, market, enable_report=False)
    
    stock_code = "AAPL.O"
    
    # 测试1: 回测未开始时访问数据应该失败
    logger.info("\n测试1: 回测未开始时访问数据")
    try:
        engine.get_price(stock_code)
        logger.error("应该抛出异常，但没有")
    except ValueError as e:
        logger.info(f"正确抛出异常: {e}")
    
    # 测试2: 运行回测并测试数据访问
    logger.info("\n测试2: 运行回测并测试数据访问")
    
    def on_trading_day(eng: BacktestEngine, date: str):
        """测试回调函数"""
        if eng.date_index == 0:  # 第一天
            logger.info(f"\n[{date}] 测试数据访问:")
            
            # 测试价格访问
            price = eng.get_price(stock_code)
            high = eng.get_high_price(stock_code)
            low = eng.get_low_price(stock_code)
            open_price = eng.get_open_price(stock_code)
            volume = eng.get_volume(stock_code)
            
            logger.info(f"  价格: {price:.2f}")
            logger.info(f"  最高: {high:.2f}")
            logger.info(f"  最低: {low:.2f}")
            logger.info(f"  开盘: {open_price:.2f}")
            logger.info(f"  成交量: {volume:,.0f}")
            
            # 测试特征访问
            try:
                ret_1d = eng.get_feature("ret_1d", stock_code)
                logger.info(f"  特征 ret_1d: {ret_1d}")
            except Exception as e:
                logger.warning(f"  特征 ret_1d 获取失败: {e}")
            
            # 测试批量获取特征
            try:
                features = eng.get_features(["ret_1d", "ret_5d"], stock_code)
                logger.info(f"  批量特征: {features}")
            except Exception as e:
                logger.warning(f"  批量特征获取失败: {e}")
            
            # 测试访问未来数据应该失败
            logger.info(f"\n  测试日期保护:")
            try:
                future_date = "2099-12-31"
                eng.get_price(stock_code, future_date)
                logger.error("  应该抛出异常，但没有")
            except ValueError as e:
                logger.info(f"  正确阻止访问未来数据: {e}")
            
            # 测试访问历史数据应该成功
            try:
                # 访问当前日期应该成功
                current_price = eng.get_price(stock_code, date)
                logger.info(f"  访问当前日期数据成功: {current_price:.2f}")
            except Exception as e:
                logger.error(f"  访问当前日期数据失败: {e}")
    
    engine.on_date(on_trading_day)
    
    # 运行回测（只运行几天）
    available_dates = market.get_available_dates(stock_code)
    if available_dates:
        start_date = available_dates[0]
        end_date = available_dates[min(4, len(available_dates) - 1)]  # 只运行5天
        engine.run(stock_code, start_date=start_date, end_date=end_date)
    
    logger.info("")
    for line in log_section("测试完成"):
        logger.info(line)


if __name__ == "__main__":
    test_engine_data_access()

