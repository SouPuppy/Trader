"""
基于 LLM 的 DCA 策略回测示例（带不确定性风险控制）
使用 LLMDCAAgentWithRiskControl 实现带 LLM 不确定性评估的风险控制定投策略
投资策略与 DCA 完全一致（每月固定金额买入），但增加了 LLM 不确定性评估和风险控制
"""
import sys
from pathlib import Path

# 添加项目根目录到 Python 路径
project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from trader.agent.agent_dca_with_llm_gate import DCAAgentWithLLMGate
from trader.backtest.account import Account
from trader.backtest.market import Market
from trader.backtest.engine import BacktestEngine
from trader.logger import get_logger, log_separator, log_section

logger = get_logger(__name__)


def llm_gate_strategy(
    stock_code: str = "AAPL.O",
    initial_cash: float = 1000000.0,
    monthly_investment: float = 100000.0,
    llm_model: str = "deepseek-chat",
    llm_temperature: float = 0.3,
    start_date: str = None,
    end_date: str = None
):
    """
    带 LLM Gate 的 DCA 策略回测（使用 DCAAgentWithLLMGate）
    
    投资策略与 DCA 完全一致（每月固定金额买入），但增加了：
    1. LLM Gate 评估：每次定投前，使用 LLM 评估是否应该执行
    2. 如果 LLM 返回 should_execute=false，跳过本次定投
    
    Args:
        stock_code: 股票代码
        initial_cash: 初始资金（元）
        monthly_investment: 每月定投金额（元）
        llm_model: LLM 模型名称
        llm_temperature: LLM 温度参数（控制随机性）
        start_date: 开始日期（格式: YYYY-MM-DD），如果为 None 则从最早可用日期开始
        end_date: 结束日期（格式: YYYY-MM-DD），如果为 None 则到最新日期
    """
    for line in log_section("带 LLM Gate 的 DCA 策略回测"):
        logger.info(line)
    logger.info(f"股票代码: {stock_code}")
    logger.info(f"初始资金: {initial_cash:,.2f} 元")
    logger.info(f"每月定投: {monthly_investment:,.2f} 元")
    logger.info(f"LLM 模型: {llm_model}")
    logger.info(f"LLM 温度: {llm_temperature}")
    
    # 初始化市场、账户和回测引擎
    market = Market(price_adjustment=0.01)
    account = Account(initial_cash=initial_cash)
    
    # 生成报告标题（包含策略名称和参数，使用英文）
    report_title = (
        f"LLM_Gate_DCA_Strategy_{stock_code}_"
        f"monthly{int(monthly_investment)}_"
        f"llmTemp{llm_temperature:.2f}"
    )
    engine = BacktestEngine(account, market, report_title=report_title)
    
    # 获取可用日期（用于显示数据范围）
    available_dates = market.get_available_dates(stock_code)
    if not available_dates:
        logger.error(f"未找到股票 {stock_code} 的数据")
        return
    
    logger.info(f"数据范围: {available_dates[0]} 至 {available_dates[-1]}")
    
    # 创建带 LLM Gate 的 DCA Agent
    agent = DCAAgentWithLLMGate(
        name="LLM_Gate_DCA_Strategy",
        monthly_investment=monthly_investment,
        dca_frequency="monthly",
        llm_model=llm_model,
        llm_temperature=llm_temperature
    )
    
    # 设置定投的股票代码
    agent.set_dca_stocks([stock_code])
    
    # 注册交易日回调：调用 agent 的 on_date 方法执行策略
    def on_trading_day(eng: BacktestEngine, date: str):
        """每个交易日的回调"""
        agent.on_date(eng, date)
    
    engine.on_date(on_trading_day)
    
    # 运行回测
    logger.info("")
    logger.info("开始回测...")
    logger.info("")
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
    
    # 获取 Gate 统计信息
    gate_stats = agent.get_gate_stats()
    
    # 输出结果
    logger.info("")
    for line in log_section("回测结果"):
        logger.info(line)
    logger.info(f"定投次数: {agent.investment_count}")
    logger.info(f"LLM Gate 通过次数: {gate_stats['gate_passed_count']}")
    logger.info(f"LLM Gate 跳过次数: {gate_stats['gate_skipped_count']}")
    logger.info(f"LLM Gate 总评估次数: {gate_stats['total_evaluations']}")
    logger.info(f"初始资金: {initial_cash:,.2f} 元")
    logger.info(f"实际投入: {total_invested:,.2f} 元")
    logger.info(f"当前现金: {account.cash:,.2f} 元")
    logger.info(f"最终权益: {equity:,.2f} 元")
    logger.info(f"总盈亏: {profit:+,.2f} 元")
    logger.info(f"总收益率: {return_pct:+.2f}%")
    logger.info(f"交易次数: {len(account.trades)}")
    logger.info(f"持仓数量: {len(account.positions)}")
    
    # 输出持仓明细
    if account.positions:
        logger.info("")
        logger.info("持仓明细:")
        for symbol, position in account.positions.items():
            current_price = market_prices.get(symbol, position["average_price"])
            position_value = position["shares"] * current_price
            position_weight = position_value / equity if equity > 0 else 0.0
            profit_per_share = current_price - position["average_price"]
            total_profit = profit_per_share * position["shares"]
            
            logger.info(
                f"  {symbol}: {position['shares']} 股 @ {position['average_price']:.2f}, "
                f"现价 {current_price:.2f}, "
                f"市值 {position_value:,.2f} ({position_weight*100:.1f}%), "
                f"盈亏 {total_profit:+,.2f}"
            )
    
    logger.info(log_separator())
    
    # 输出详细账户摘要
    logger.info("")
    logger.info(account.summary(market_prices))


if __name__ == "__main__":
    # 执行带 LLM Gate 的 DCA 策略回测
    # 
    # 策略说明：
    # - 基于 DCA 策略（每月固定金额买入）
    # - 每次定投前，使用 LLM 评估是否应该执行
    # - LLM 返回 should_execute=false 时，跳过本次定投
    # - 这样可以避免在市场不利时执行定投
    llm_gate_strategy(
        stock_code="AAPL.O",
        initial_cash=1000000.0,
        monthly_investment=100000.0,  # 每月定投金额，与 DCA 策略一致
        llm_model="deepseek-chat",
        llm_temperature=0.3,  # 较低温度以获得更一致的结果
        start_date="2023-01-03",  # 可以设置为 None 从最早日期开始
        end_date="2023-12-29"  # 可以设置为 None 到最新日期
    )

