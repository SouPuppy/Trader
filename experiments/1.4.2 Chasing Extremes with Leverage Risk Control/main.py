"""
Chasing Extremes Agent 回测示例（带杠杆风险控制）
在 Chasing Extremes Agent 基础上添加杠杆限制风险控制，用于测试 risk control 是否有效

策略逻辑（稳定亏钱的疯狂策略 - 反向操作）：
- 反向操作：价格上涨时卖出，价格下跌时买入（总是在错误的时间交易）
- 低阈值触发：极端波动阈值降低到 1%，让它更容易触发交易
- 最小仓位保证：即使波动很小，也至少给予 50% 仓位，让它频繁交易
- 通过杠杆限制风险控制，限制总持仓市值不超过账户权益的一定比例
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
from trader.backtest.report import BacktestReport
from trader.config_loader import get_test_stocks
from trader.risk.OrderIntent import OrderIntent, OrderSide, PriceType
from trader.logger import get_logger, log_separator, log_section
from typing import List, Optional, Dict

logger = get_logger(__name__)


def chasing_extremes_with_risk_control_backtest_single(
    stock_code: str = "AAPL.O",
    initial_cash: float = 1_000_000.0,
    extreme_threshold: float = 0.01,  # 极端波动阈值（1%，降低阈值让它更容易触发）
    lookback_days: int = 1,  # 回看天数
    max_position_weight: float = 1.0,  # 最大仓位（全仓）
    chase_up: bool = True,  # 是否追涨
    chase_down: bool = True,  # 是否追跌
    max_leverage: float = 1.0,  # 最大杠杆率（1.0 = 允许全仓，总持仓市值不超过账户权益的100%）
    start_date: str = None,
    end_date: str = None
) -> Optional[Dict]:
    """
    Chasing Extremes Agent 回测（带杠杆风险控制）
    
    Args:
        stock_code: 股票代码
        initial_cash: 初始资金
        extreme_threshold: 极端波动阈值（如 0.01 表示 1%，降低阈值让它更容易触发）
        lookback_days: 回看天数，用于计算涨跌幅
        max_position_weight: 最大仓位（默认全仓）
        chase_up: 是否追涨（反向操作：价格上涨时卖出）
        chase_down: 是否追跌（反向操作：价格下跌时买入）
        max_leverage: 最大杠杆率（1.0 = 允许全仓，总持仓市值不超过账户权益的100%）
        start_date: 开始日期
        end_date: 结束日期
    """
    for line in log_section("Chasing Extremes Strategy Backtest (with Leverage Risk Control)"):
        logger.info(line)
    logger.info(f"股票代码: {stock_code}")
    logger.info(f"初始资金: {initial_cash:,.2f} 元")
    logger.info(f"极端波动阈值: {extreme_threshold*100:.1f}%")
    logger.info(f"回看天数: {lookback_days} 天")
    logger.info(f"最大仓位: {max_position_weight*100:.0f}%")
    logger.info(f"追涨: {chase_up}")
    logger.info(f"追跌: {chase_down}")
    logger.info(f"最大杠杆率: {max_leverage:.2f}")
    
    # 初始化市场、账户和回测引擎（每只股票独立的账户）
    market = Market(price_adjustment=0.01)
    account = Account(initial_cash=initial_cash)
    
    # 启用报告生成以记录每日状态，但不生成独立报告文件（通过设置 report_title=None）
    # Chasing Extremes 策略不需要训练，使用所有数据（train_test_split_ratio=0.0）
    engine = BacktestEngine(account, market, enable_report=True, report_title=None, train_test_split_ratio=0.0)
    
    # 获取可用日期
    available_dates = market.get_available_dates(stock_code)
    if not available_dates:
        logger.error(f"未找到股票 {stock_code} 的数据")
        return
    
    logger.info(f"数据范围: {available_dates[0]} 至 {available_dates[-1]}")
    
    # 创建带风险控制的 Chasing Extremes Agent
    agent = ChasingExtremesAgentWithRiskControl(
        name="ChasingExtremes_RiskControl",
        extreme_threshold=extreme_threshold,
        lookback_days=lookback_days,
        max_position_weight=max_position_weight,
        chase_up=chase_up,
        chase_down=chase_down,
        max_leverage=max_leverage
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
        # 即使变化很小，只要有信号就交易
        min_trade_amount = 1000  # 最小交易金额（降低到1000元）
        if abs(diff_value) < min_trade_amount and weight > 0:
            # 如果权重>0但金额变化小，强制买入至少最小金额
            if diff_value > 0:
                diff_value = min_trade_amount
            else:
                # 卖出至少最小金额对应的股数
                if current_price and current_price > 0:
                    diff_value = -min_trade_amount
        
        # 如果变化很小，不需要交易
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
    
    # 计算最终结果
    final_date = end_date if end_date else available_dates[-1]
    final_price = market.get_price(stock_code, final_date)
    if final_price is None:
        final_price = market.get_price(stock_code)
    
    if final_price is None:
        logger.error("无法获取最终价格")
        return None
    
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
    
    return {
        'stock_code': stock_code,
        'initial_cash': initial_cash,
        'final_equity': equity,
        'profit': profit,
        'return_pct': return_pct,
        'num_trades': len(account.trades),
        'leverage': leverage,
        'data_range': f"{available_dates[0]} → {available_dates[-1]}",
        # 保存账户和引擎信息，用于生成图表和详细指标
        'account': account,
        'engine': engine,
        'market': market,
        'start_date': start_date or available_dates[0],
        'end_date': end_date or available_dates[-1]
    }


def chasing_extremes_with_risk_control_backtest_all_stocks(
    stock_codes: Optional[List[str]] = None,
    initial_cash: float = 1_000_000.0,
    extreme_threshold: float = 0.01,
    lookback_days: int = 1,
    max_position_weight: float = 1.0,
    chase_up: bool = True,
    chase_down: bool = True,
    max_leverage: float = 0.8,
    start_date: str = None,
    end_date: str = None
):
    """
    对所有股票分别进行 Chasing Extremes 回测，生成综合报告
    """
    # 如果没有指定股票代码，从配置文件读取
    if stock_codes is None:
        stock_codes = get_test_stocks()
    
    for line in log_section("Chasing Extremes Strategy Backtest - 多股票单股票测试"):
        logger.info(line)
    logger.info(f"股票数量: {len(stock_codes)}")
    logger.info(f"股票列表: {stock_codes}")
    logger.info(f"初始资金: {initial_cash:,.2f} 元/股票")
    logger.info("注意：每只股票使用独立的账户，分别测试")
    logger.info("")
    
    # 运行所有股票的回测
    results = []
    for i, stock_code in enumerate(stock_codes, 1):
        logger.info("")
        logger.info(f"========== [{i}/{len(stock_codes)}] {stock_code} ==========")
        
        try:
            result = chasing_extremes_with_risk_control_backtest_single(
                stock_code=stock_code,
                initial_cash=initial_cash,
                extreme_threshold=extreme_threshold,
                lookback_days=lookback_days,
                max_position_weight=max_position_weight,
                chase_up=chase_up,
                chase_down=chase_down,
                max_leverage=max_leverage,
                start_date=start_date,
                end_date=end_date
            )
            
            if result:
                results.append(result)
        except Exception as e:
            logger.error(f"回测 {stock_code} 时出错: {e}", exc_info=True)
            results.append({
                'stock_code': stock_code,
                'error': str(e)
            })
    
    # 生成综合报告
    logger.info("")
    logger.info("")
    for line in log_section("综合回测报告"):
        logger.info(line)
    
    if results:
        successful_results = [r for r in results if 'error' not in r]
        
        if successful_results:
            total_profit = sum(r['profit'] for r in successful_results)
            total_return = sum(r['return_pct'] for r in successful_results)
            avg_return = total_return / len(successful_results)
            total_trades = sum(r['num_trades'] for r in successful_results)
            total_initial_cash = sum(r['initial_cash'] for r in successful_results)
            total_final_equity = sum(r['final_equity'] for r in successful_results)
            
            logger.info(f"成功回测股票数: {len(successful_results)}/{len(stock_codes)}")
            logger.info(f"总初始资金: {total_initial_cash:,.2f} 元")
            logger.info(f"总最终权益: {total_final_equity:,.2f} 元")
            logger.info(f"总盈亏: {total_profit:+,.2f} 元")
            logger.info(f"平均收益率: {avg_return:+.2f}%")
            logger.info(f"总交易次数: {total_trades}")
            logger.info("")
            
            # 详细结果表格
            logger.info("详细结果:")
            logger.info(f"{'股票代码':<15} {'初始资金':>15} {'最终权益':>15} {'盈亏':>15} {'收益率':>10} {'交易次数':>8}")
            logger.info("-" * 90)
            
            for r in successful_results:
                logger.info(
                    f"{r['stock_code']:<15} "
                    f"{r['initial_cash']:>15,.2f} "
                    f"{r['final_equity']:>15,.2f} "
                    f"{r['profit']:>+15,.2f} "
                    f"{r['return_pct']:>+10.2f}% "
                    f"{r['num_trades']:>8}"
                )
            
            # 失败的结果
            failed_results = [r for r in results if 'error' in r]
            if failed_results:
                logger.info("")
                logger.info("失败的回测:")
                for r in failed_results:
                    logger.error(f"{r['stock_code']}: {r['error']}")
            
            # 生成综合报告文件
            logger.info("")
            logger.info("生成综合报告文件...")
            
            # 从结果中提取实际的日期范围
            actual_start_date = start_date or "N/A"
            actual_end_date = end_date or "N/A"
            if successful_results and 'data_range' in successful_results[0]:
                data_range = successful_results[0]['data_range']
                if ' → ' in data_range:
                    parts = data_range.split(' → ')
                    if not start_date and len(parts) > 0:
                        actual_start_date = parts[0].strip()
                    if not end_date and len(parts) > 1:
                        actual_end_date = parts[1].strip()
            
            # 获取实验文件夹名称作为输出目录名称
            experiment_dir = Path(__file__).parent.name
            from trader.config import PROJECT_ROOT
            output_dir = PROJECT_ROOT / 'output' / 'backtest' / experiment_dir
            report = BacktestReport(output_dir=output_dir, title=None)
            
            strategy_params = {
                "extreme_threshold": f"{extreme_threshold*100:.1f}%",
                "lookback_days": lookback_days,
                "max_position_weight": f"{max_position_weight*100:.0f}%",
                "chase_up": chase_up,
                "chase_down": chase_down,
                "max_leverage": f"{max_leverage:.2f}"
            }
            
            report_file = report.generate_multi_stock_report(
                results=results,
                strategy_name="Chasing Extremes with Leverage Risk Control",
                start_date=actual_start_date,
                end_date=actual_end_date,
                initial_cash_per_stock=initial_cash,
                strategy_params=strategy_params
            )
            logger.info(f"综合报告已保存: {report_file}")
        else:
            logger.error("所有回测都失败了")
    else:
        logger.error("没有回测结果")
    
    logger.info(log_separator())


if __name__ == "__main__":
    # 对所有股票分别进行 Chasing Extremes 回测，生成综合报告
    chasing_extremes_with_risk_control_backtest_all_stocks(
        stock_codes=None,  # 从 trader/config.toml 读取
        initial_cash=1_000_000.0,
        extreme_threshold=0.01,  # 1% 极端波动阈值（降低阈值，让它更容易触发）
        lookback_days=1,  # 回看1天
        max_position_weight=1.0,  # 全仓
        chase_up=True,  # 反向操作：价格上涨时卖出
        chase_down=True,  # 反向操作：价格下跌时买入
        max_leverage=1.0,  # 最大杠杆率（1.0 = 允许全仓，总持仓市值不超过账户权益的100%）
        start_date="2023-01-01",
        end_date="2023-12-31"
    )

