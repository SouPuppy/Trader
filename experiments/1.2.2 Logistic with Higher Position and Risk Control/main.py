"""
Logistic Regression 逻辑回归策略回测示例（增加投资比例 + 风险控制）
使用 LogisticAgentWithRiskControl 实现基于逻辑回归的预测策略，增加投资比例并通过杠杆限制进行风险控制
"""
import sys
from pathlib import Path

# 添加项目根目录到 Python 路径
project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from trader.agent.agent_logistic_with_risk_control import LogisticAgentWithRiskControl
from trader.backtest.account import Account
from trader.backtest.market import Market
from trader.backtest.engine import BacktestEngine
from trader.risk.OrderIntent import OrderIntent, OrderSide, PriceType
from trader.logger import get_logger, log_separator, log_section

logger = get_logger(__name__)


def logistic_strategy_with_risk_control(
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
    max_leverage: float = 0.8,  # 最大杠杆率（0.8 = 总持仓市值不超过账户权益的80%，让风险控制起作用）
    start_date: str = None,
    end_date: str = None
):
    """
    逻辑回归策略回测（使用 LogisticAgentWithRiskControl，增加投资比例 + 风险控制）
    
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
        max_leverage: 最大杠杆率（传递给 LeverageLimitRiskManager）
        start_date: 开始日期
        end_date: 结束日期
    """
    for line in log_section("逻辑回归策略回测（增加投资比例 + 风险控制）"):
        logger.info(line)
    logger.info(f"股票代码: {stock_code}")
    logger.info(f"初始资金: {initial_cash:,.2f} 元")
    logger.info(f"训练窗口: {train_window_days} 天")
    logger.info(f"预测周期: {prediction_horizon} 天")
    logger.info(f"收益阈值: {ret_threshold:.2%}")
    logger.info(f"重新训练频率: 每 {retrain_frequency} 个交易日")
    logger.info(f"单股最大仓位: {max_position_weight*100:.0f}%")
    logger.info(f"总仓位上限: {max_total_weight*100:.0f}%")
    logger.info(f"最大杠杆率: {max_leverage:.2f}")
    logger.info(f"训练/测试分割比例: {train_test_split_ratio*100:.0f}% / {(1-train_test_split_ratio)*100:.0f}%")
    
    # 初始化市场、账户和回测引擎
    market = Market(price_adjustment=0.01)
    account = Account(initial_cash=initial_cash)
    
    # 生成报告标题（包含策略名称和参数）
    report_title = (
        f"Logistic_RiskControl_Strategy_{stock_code}_"
        f"train{train_window_days}_horizon{prediction_horizon}_"
        f"retrain{retrain_frequency}_maxPos{max_position_weight*100:.0f}_"
        f"maxLev{max_leverage:.2f}"
    )
    engine = BacktestEngine(
        account, market, 
        report_title=report_title,
        train_test_split_ratio=train_test_split_ratio
    )
    
    # 获取可用日期
    available_dates = market.get_available_dates(stock_code)
    if not available_dates:
        logger.error(f"未找到股票 {stock_code} 的数据")
        return
    
    logger.info(f"数据范围: {available_dates[0]} 至 {available_dates[-1]}")
    
    # 创建带风险控制的逻辑回归 Agent
    agent = LogisticAgentWithRiskControl(
        name="Logistic_RiskControl_Strategy",
        feature_names=feature_names,
        train_window_days=train_window_days,
        prediction_horizon=prediction_horizon,
        ret_threshold=ret_threshold,
        retrain_frequency=retrain_frequency,
        max_position_weight=max_position_weight,
        min_score_threshold=min_score_threshold,
        max_total_weight=max_total_weight,
        train_test_split_ratio=train_test_split_ratio,
        max_leverage=max_leverage
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
        
        # 如果权重为0，不需要交易
        if weight == 0.0:
            # 检查是否需要卖出（如果当前有持仓但权重为0）
            position = account.get_position(stock_code)
            if position and position['shares'] > 0:
                # 卖出所有持仓
                eng.sell(stock_code, shares=position['shares'])
            return
        
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
        
        # 如果变化很小，不需要交易
        if abs(diff_value) < 100:
            return
        
        # 生成订单意图
        order_intents = []
        if diff_value > 0:
            # 买入
            order_intent = OrderIntent(
                symbol=stock_code,
                side=OrderSide.BUY,
                timestamp=date,
                target_weight=weight,  # 使用目标权重
                price_type=PriceType.MKT,
                agent_name=agent.name,
                confidence=abs(score),  # 使用 score 的绝对值作为置信度
                metadata={"score": score, "diff_value": diff_value}
            )
            order_intents.append(order_intent)
        else:
            # 卖出
            shares_to_sell = int(abs(diff_value) / current_price) if current_price else 0
            if shares_to_sell > 0 and position:
                # 不能卖出超过持仓数量
                shares_to_sell = min(shares_to_sell, position['shares'])
                if shares_to_sell > 0:
                    order_intent = OrderIntent(
                        symbol=stock_code,
                        side=OrderSide.SELL,
                        timestamp=date,
                        qty=shares_to_sell,
                        price_type=PriceType.MKT,
                        agent_name=agent.name,
                        confidence=1.0,
                        metadata={"score": score, "diff_value": diff_value}
                    )
                    order_intents.append(order_intent)
        
        # 应用风险控制（只对买入订单应用）
        if order_intents:
            approved_orders = agent.apply_risk_control(order_intents, eng)
            
            # 执行订单
            agent.execute_orders(approved_orders, eng)
    
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
    
    # 计算杠杆率
    total_position_value = sum(
        position["shares"] * market_prices.get(symbol, position["average_price"])
        for symbol, position in account.positions.items()
    )
    leverage = total_position_value / equity if equity > 0 else 0.0
    
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
    logger.info(f"当前杠杆率: {leverage:.2f} (上限: {max_leverage:.2f})")
    logger.info(log_separator())
    
    # 输出详细账户摘要
    logger.info("")
    logger.info(account.summary(market_prices))


if __name__ == "__main__":
    # 执行逻辑回归策略（增加投资比例 + 风险控制）
    logistic_strategy_with_risk_control(
        stock_code="AAPL.O",
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
        max_leverage=0.8,  # 最大杠杆率（0.8 = 总持仓市值不超过账户权益的80%，让风险控制起作用）
        start_date=None,  # 从最早日期开始
        end_date=None     # 到最新日期
    )

