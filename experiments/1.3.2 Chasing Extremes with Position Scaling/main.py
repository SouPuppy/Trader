"""
Chasing Extremes Agent 回测示例（带 Position Scaling 风险控制）
在 Chasing Extremes Agent 基础上添加 Position Scaling 风险控制，用于测试 risk control 是否有效

策略逻辑（稳定亏钱的疯狂策略 - 反向操作）：
- 反向操作：价格上涨时卖出，价格下跌时买入（总是在错误的时间交易）
- 低阈值触发：极端波动阈值降低到 1%，让它更容易触发交易
- 最小仓位保证：即使波动很小，也至少给予 50% 仓位，让它频繁交易
- Position Scaling 风险控制：根据账户权益、置信度、历史表现动态调整仓位
"""
import sys
from pathlib import Path

# 添加项目根目录到 Python 路径
project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from trader.agent.agent_chasing_extremes_with_risk_control import ChasingExtremesAgentWithRiskControl
from trader.backtest.account import Account
from trader.backtest.market import Market
from trader.backtest.engine import BacktestEngine
from trader.risk.OrderIntent import OrderIntent, OrderSide, PriceType
from trader.risk.RiskManager import RiskManagerPipeline
from trader.risk.control_position_scaling import PositionScalingRiskManager
from trader.logger import get_logger, log_separator, log_section

logger = get_logger(__name__)


class ChasingExtremesAgentWithPositionScaling(ChasingExtremesAgentWithRiskControl):
    """
    带 Position Scaling 风险控制的 Chasing Extremes Agent
    """
    
    def __init__(
        self,
        name: str = "ChasingExtremesAgentWithPositionScaling",
        extreme_threshold: float = 0.01,
        lookback_days: int = 1,
        max_position_weight: float = 1.0,
        min_score_threshold: float = 0.0,
        max_total_weight: float = 1.0,
        chase_up: bool = True,
        chase_down: bool = True,
        # Position Scaling 参数
        enable_equity_scaling: bool = True,
        base_equity: float = 1_000_000.0,
        equity_scaling_factor: float = 0.5,
        enable_confidence_scaling: bool = True,
        confidence_power: float = 1.0,
        enable_performance_scaling: bool = True,
        consecutive_loss_threshold: int = 3,
        consecutive_win_threshold: int = 3,
        loss_scaling_factor: float = 0.5,
        win_scaling_factor: float = 1.2,
    ):
        """
        初始化带 Position Scaling 风险控制的 Chasing Extremes Agent
        
        Args:
            name: Agent 名称
            extreme_threshold: 极端波动阈值
            lookback_days: 回看天数
            max_position_weight: 最大仓位
            min_score_threshold: 最小 score 阈值
            max_total_weight: 总配置比例上限
            chase_up: 是否追涨（反向操作：价格上涨时卖出）
            chase_down: 是否追跌（反向操作：价格下跌时买入）
            enable_equity_scaling: 是否启用账户权益缩放
            base_equity: 基准账户权益
            equity_scaling_factor: 权益缩放因子
            enable_confidence_scaling: 是否启用置信度缩放
            confidence_power: 置信度幂次
            enable_performance_scaling: 是否启用历史表现缩放
            consecutive_loss_threshold: 连续亏损阈值
            consecutive_win_threshold: 连续盈利阈值
            loss_scaling_factor: 连续亏损后的缩放因子
            win_scaling_factor: 连续盈利后的缩放因子
        """
        # 先调用父类初始化（不传 max_leverage，因为我们用 Position Scaling）
        super().__init__(
            name=name,
            extreme_threshold=extreme_threshold,
            lookback_days=lookback_days,
            max_position_weight=max_position_weight,
            min_score_threshold=min_score_threshold,
            max_total_weight=max_total_weight,
            chase_up=chase_up,
            chase_down=chase_down,
            max_leverage=1.0  # 不使用杠杆限制，使用 Position Scaling
        )
        
        # 替换风险管道，使用 Position Scaling
        self.risk_pipeline = RiskManagerPipeline([
            PositionScalingRiskManager(
                name="PositionScaling",
                enable_equity_scaling=enable_equity_scaling,
                base_equity=base_equity,
                equity_scaling_factor=equity_scaling_factor,
                enable_confidence_scaling=enable_confidence_scaling,
                confidence_power=confidence_power,
                enable_performance_scaling=enable_performance_scaling,
                consecutive_loss_threshold=consecutive_loss_threshold,
                consecutive_win_threshold=consecutive_win_threshold,
                loss_scaling_factor=loss_scaling_factor,
                win_scaling_factor=win_scaling_factor,
            )
        ])


def chasing_extremes_with_position_scaling_backtest(
    stock_code: str = "AAPL.O",
    initial_cash: float = 1_000_000.0,
    extreme_threshold: float = 0.01,
    lookback_days: int = 1,
    max_position_weight: float = 1.0,
    chase_up: bool = True,
    chase_down: bool = True,
    # Position Scaling 参数
    enable_equity_scaling: bool = True,
    base_equity: float = 1_000_000.0,
    equity_scaling_factor: float = 0.5,
    enable_confidence_scaling: bool = True,
    confidence_power: float = 1.0,
    enable_performance_scaling: bool = True,
    consecutive_loss_threshold: int = 3,
    consecutive_win_threshold: int = 3,
    loss_scaling_factor: float = 0.5,
    win_scaling_factor: float = 1.2,
    start_date: str = None,
    end_date: str = None
):
    """
    Chasing Extremes Agent 回测（带 Position Scaling 风险控制）
    """
    for line in log_section("Chasing Extremes Strategy Backtest (with Position Scaling)"):
        logger.info(line)
    logger.info(f"股票代码: {stock_code}")
    logger.info(f"初始资金: {initial_cash:,.2f} 元")
    logger.info(f"极端波动阈值: {extreme_threshold*100:.1f}%")
    logger.info(f"回看天数: {lookback_days} 天")
    logger.info(f"最大仓位: {max_position_weight*100:.0f}%")
    logger.info(f"追涨: {chase_up} (反向操作：价格上涨时卖出)")
    logger.info(f"追跌: {chase_down} (反向操作：价格下跌时买入)")
    logger.info(f"Position Scaling:")
    logger.info(f"  - 账户权益缩放: {enable_equity_scaling} (基准: {base_equity:,.0f}, 因子: {equity_scaling_factor})")
    logger.info(f"  - 置信度缩放: {enable_confidence_scaling} (幂次: {confidence_power})")
    logger.info(f"  - 历史表现缩放: {enable_performance_scaling} (亏损阈值: {consecutive_loss_threshold}, 盈利阈值: {consecutive_win_threshold})")
    
    # 初始化市场、账户和回测引擎
    market = Market(price_adjustment=0.01)
    account = Account(initial_cash=initial_cash)
    
    # 生成报告标题
    report_title = (
        f"ChasingExtremes_PositionScaling_{stock_code}_"
        f"threshold{extreme_threshold*100:.0f}pct_"
        f"lookback{lookback_days}_"
        f"maxPos{max_position_weight*100:.0f}"
    )
    engine = BacktestEngine(account, market, report_title=report_title)
    
    # 获取可用日期
    available_dates = market.get_available_dates(stock_code)
    if not available_dates:
        logger.error(f"未找到股票 {stock_code} 的数据")
        return
    
    logger.info(f"数据范围: {available_dates[0]} 至 {available_dates[-1]}")
    
    # 创建带 Position Scaling 风险控制的 Chasing Extremes Agent
    agent = ChasingExtremesAgentWithPositionScaling(
        name="ChasingExtremes_PositionScaling",
        extreme_threshold=extreme_threshold,
        lookback_days=lookback_days,
        max_position_weight=max_position_weight,
        chase_up=chase_up,
        chase_down=chase_down,
        enable_equity_scaling=enable_equity_scaling,
        base_equity=base_equity,
        equity_scaling_factor=equity_scaling_factor,
        enable_confidence_scaling=enable_confidence_scaling,
        confidence_power=confidence_power,
        enable_performance_scaling=enable_performance_scaling,
        consecutive_loss_threshold=consecutive_loss_threshold,
        consecutive_win_threshold=consecutive_win_threshold,
        loss_scaling_factor=loss_scaling_factor,
        win_scaling_factor=win_scaling_factor,
    )
    
    # 注册交易日回调：执行策略
    def on_trading_day(eng: BacktestEngine, date: str):
        """每个交易日的回调"""
        # 更新 agent 状态
        agent.on_date(eng, date)
        
        # 获取当前股票的 score
        score = agent.score(stock_code, eng)
        
        # 计算权重
        weight = agent.weight(stock_code, score, eng)
        
        # 如果权重为0，检查是否需要卖出
        if weight == 0.0:
            position = account.get_position(stock_code)
            if position and position['shares'] > 0:
                # 如果 score < 0（反向操作：价格上涨时卖出），卖出所有持仓
                if score < 0:
                    eng.sell(stock_code, shares=position['shares'])
                    logger.info(f"[{date}] 反向操作卖出 {stock_code}: {position['shares']} 股 (score={score:.3f})")
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
        
        # 降低交易阈值，让它更频繁地交易（稳定亏钱）
        min_trade_amount = 1000  # 最小交易金额
        if abs(diff_value) < min_trade_amount and weight > 0:
            if diff_value > 0:
                diff_value = min_trade_amount
            else:
                if current_price and current_price > 0:
                    diff_value = -min_trade_amount
        
        if abs(diff_value) < min_trade_amount:
            return
        
        # 生成订单意图
        order_intents = []
        if diff_value > 0:
            # 买入
            order_intent = OrderIntent(
                symbol=stock_code,
                side=OrderSide.BUY,
                timestamp=date,
                target_weight=weight,
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
        
        # 应用风险控制
        if order_intents:
            approved_orders = agent.apply_risk_control(order_intents, eng)
            
            # 执行订单
            agent.execute_orders(approved_orders, eng)
    
    engine.on_date(on_trading_day)
    
    # 运行回测
    logger.info("")
    logger.info("开始回测...")
    engine.run(stock_code, start_date=start_date, end_date=end_date)
    
    # 计算最终结果
    final_date = end_date if end_date else available_dates[-1]
    final_price = market.get_price(stock_code, final_date)
    if final_price is None:
        final_price = market.get_price(stock_code)
    
    if final_price is None:
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
    logger.info(log_separator())
    
    # 输出详细账户摘要
    logger.info("")
    logger.info(account.summary(market_prices))


if __name__ == "__main__":
    # 执行 Chasing Extremes 策略回测（带 Position Scaling 风险控制）
    chasing_extremes_with_position_scaling_backtest(
        stock_code="AAPL.O",
        initial_cash=1_000_000.0,
        extreme_threshold=0.01,  # 1% 极端波动阈值
        lookback_days=1,  # 回看1天
        max_position_weight=1.0,  # 全仓
        chase_up=True,  # 反向操作：价格上涨时卖出
        chase_down=True,  # 反向操作：价格下跌时买入
        enable_equity_scaling=True,  # 启用账户权益缩放
        base_equity=1_000_000.0,  # 基准账户权益
        equity_scaling_factor=0.5,  # 权益缩放因子
        enable_confidence_scaling=True,  # 启用置信度缩放
        confidence_power=1.0,  # 置信度幂次（线性）
        enable_performance_scaling=True,  # 启用历史表现缩放
        consecutive_loss_threshold=3,  # 连续亏损阈值
        consecutive_win_threshold=3,  # 连续盈利阈值
        loss_scaling_factor=0.5,  # 连续亏损后的缩放因子
        win_scaling_factor=1.2,  # 连续盈利后的缩放因子（上限 1.0）
        start_date="2023-01-01",
        end_date="2023-12-31"
    )

