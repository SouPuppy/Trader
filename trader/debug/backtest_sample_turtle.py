"""
海龟策略（Turtle Trading Strategy）回测示例
经典的趋势跟踪策略，使用突破系统和ATR进行风险管理
"""
import sys
from pathlib import Path

# 添加项目根目录到 Python 路径
project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from datetime import datetime
from typing import Dict, Optional, List
import pandas as pd
from trader.backtest.account import Account
from trader.backtest.market import Market
from trader.backtest.engine import BacktestEngine
from trader.logger import get_logger

logger = get_logger(__name__)


class TurtleStrategy:
    """海龟策略"""
    
    def __init__(
        self,
        entry_period: int = 20,  # 突破周期（N日最高/最低）
        exit_period: int = 10,   # 退出周期
        atr_period: int = 20,    # ATR计算周期
        risk_per_trade: float = 0.02,  # 每次交易风险（账户资金的百分比）
        stop_loss_atr: float = 2.0,   # 止损距离（ATR倍数）
        max_positions: int = 4,       # 最大加仓次数
        add_position_atr: float = 0.5  # 加仓距离（ATR倍数）
    ):
        """
        初始化海龟策略
        
        Args:
            entry_period: 突破周期，用于计算N日最高/最低价
            exit_period: 退出周期，用于计算退出信号
            atr_period: ATR计算周期
            risk_per_trade: 每次交易的风险（账户资金的百分比）
            stop_loss_atr: 止损距离（ATR的倍数）
            max_positions: 最大加仓次数
            add_position_atr: 加仓距离（ATR的倍数）
        """
        self.entry_period = entry_period
        self.exit_period = exit_period
        self.atr_period = atr_period
        self.risk_per_trade = risk_per_trade
        self.stop_loss_atr = stop_loss_atr
        self.max_positions = max_positions
        self.add_position_atr = add_position_atr
        
        # 存储历史数据
        self.price_history: List[Dict] = []
        self.entry_price: Optional[float] = None  # 入场价格
        self.stop_loss: Optional[float] = None   # 止损价格
        self.add_position_levels: List[float] = []  # 加仓价格水平
        self.position_count: int = 0  # 当前加仓次数
    
    def calculate_atr(self, df: pd.DataFrame, period: int = 20) -> float:
        """
        计算ATR（Average True Range）
        
        Args:
            df: 价格数据DataFrame，需要包含 high_price, low_price, prev_close
            period: ATR计算周期
            
        Returns:
            float: ATR值
        """
        if len(df) < period:
            return 0.0
        
        # 计算True Range
        df = df.copy()
        df['tr1'] = df['high_price'] - df['low_price']
        df['tr2'] = abs(df['high_price'] - df['prev_close'].shift(1))
        df['tr3'] = abs(df['low_price'] - df['prev_close'].shift(1))
        df['tr'] = df[['tr1', 'tr2', 'tr3']].max(axis=1)
        
        # 计算ATR（简单移动平均）
        atr = df['tr'].tail(period).mean()
        return float(atr) if pd.notna(atr) else 0.0
    
    def update_price_history(self, date: str, high: float, low: float, 
                            close: float, prev_close: float):
        """更新价格历史"""
        self.price_history.append({
            'date': date,
            'high_price': high,
            'low_price': low,
            'close_price': close,
            'prev_close': prev_close
        })
        
        # 只保留必要的历史数据（最多保留entry_period + atr_period天）
        max_history = max(self.entry_period, self.exit_period, self.atr_period) + 10
        if len(self.price_history) > max_history:
            self.price_history = self.price_history[-max_history:]
    
    def check_buy_signal(self, high_price: float, atr: float) -> bool:
        """检查买入信号（使用最高价）"""
        if len(self.price_history) < self.entry_period:
            return False
        
        if self.entry_price is not None:
            return False  # 已有持仓
        
        df = pd.DataFrame(self.price_history)
        # 使用前N-1天的最高价（不包括当天），这是海龟策略的标准做法
        # tail(entry_period) 会包含最后entry_period天的数据，但我们需要排除最后一天（当天）
        if len(df) >= self.entry_period + 1:
            # 排除最后一条（当天），取前N天
            entry_high = df['high_price'].iloc[-(self.entry_period+1):-1].max()
        else:
            # 如果数据不够，使用前N天（不包括最后一条）
            entry_high = df['high_price'].head(-1).tail(self.entry_period).max() if len(df) > self.entry_period else df['high_price'].head(-1).max()
        
        if high_price > entry_high:
            # 触发买入
            self.entry_price = high_price
            self.stop_loss = high_price - (atr * self.stop_loss_atr) if atr > 0 else high_price * 0.95
            self.position_count = 1
            self.add_position_levels = []
            # 计算加仓价格水平
            if atr > 0:
                for i in range(1, self.max_positions):
                    level = high_price + (atr * self.add_position_atr * i)
                    self.add_position_levels.append(level)
            return True
        return False
    
    def check_sell_signal(self, low_price: float, atr: float) -> bool:
        """检查卖出信号（使用最低价）"""
        if self.entry_price is None:
            return False  # 无持仓
        
        df = pd.DataFrame(self.price_history)
        # 使用前N-1天的最低价（不包括当天）
        if len(df) >= self.exit_period + 1:
            exit_low = df['low_price'].iloc[-(self.exit_period+1):-1].min()
        else:
            exit_low = df['low_price'].head(-1).tail(self.exit_period).min() if len(df) > self.exit_period else df['low_price'].head(-1).min()
        
        # 卖出信号：跌破退出周期最低价或触发止损
        if low_price < exit_low or (self.stop_loss and low_price < self.stop_loss):
            self.entry_price = None
            self.stop_loss = None
            self.add_position_levels = []
            self.position_count = 0
            return True
        return False
    
    def check_add_position_signal(self, high_price: float, atr: float) -> bool:
        """检查加仓信号"""
        if self.entry_price is None:
            return False  # 无持仓
        
        if not self.add_position_levels:
            return False
        
        if self.position_count >= self.max_positions:
            return False
        
        if high_price >= self.add_position_levels[0]:
            # 触发加仓
            self.position_count += 1
            # 更新止损和加仓水平
            if atr > 0:
                self.stop_loss = high_price - (atr * self.stop_loss_atr)
            self.add_position_levels.pop(0)
            return True
        return False
    
    def calculate_position_size(self, account_equity: float, atr: float, 
                               current_price: float) -> int:
        """
        计算仓位大小（基于ATR和风险）
        
        Args:
            account_equity: 账户总权益
            atr: ATR值
            current_price: 当前价格
            
        Returns:
            int: 买入股数
        """
        if atr == 0 or current_price == 0:
            return 0
        
        # 风险金额 = 账户权益 * 风险比例
        risk_amount = account_equity * self.risk_per_trade
        
        # 每股风险 = ATR * 止损倍数
        risk_per_share = atr * self.stop_loss_atr
        
        if risk_per_share == 0:
            return 0
        
        # 仓位大小 = 风险金额 / 每股风险
        shares = int(risk_amount / risk_per_share)
        
        return max(0, shares)


def turtle_strategy(
    stock_code: str = "AAPL.O",
    initial_cash: float = 100000.0,  # 海龟策略需要更多资金
    entry_period: int = 20,
    exit_period: int = 10,
    atr_period: int = 20,
    risk_per_trade: float = 0.02,
    stop_loss_atr: float = 2.0,
    max_positions: int = 4,
    add_position_atr: float = 0.5,
    start_date: str = None,
    end_date: str = None
):
    """
    海龟策略回测
    
    Args:
        stock_code: 股票代码
        initial_cash: 初始资金
        entry_period: 突破周期
        exit_period: 退出周期
        atr_period: ATR计算周期
        risk_per_trade: 每次交易风险（账户资金的百分比）
        stop_loss_atr: 止损距离（ATR倍数）
        max_positions: 最大加仓次数
        add_position_atr: 加仓距离（ATR倍数）
        start_date: 开始日期
        end_date: 结束日期
    """
    logger.info("=" * 60)
    logger.info("海龟策略回测")
    logger.info("=" * 60)
    logger.info(f"股票代码: {stock_code}")
    logger.info(f"初始资金: {initial_cash:,.2f} 元")
    logger.info(f"突破周期: {entry_period} 天")
    logger.info(f"退出周期: {exit_period} 天")
    logger.info(f"ATR周期: {atr_period} 天")
    logger.info(f"风险比例: {risk_per_trade*100:.1f}%")
    logger.info(f"止损距离: {stop_loss_atr} ATR")
    logger.info(f"最大加仓次数: {max_positions}")
    
    # 初始化市场、账户和回测引擎
    market = Market(price_adjustment=0.01)
    account = Account(initial_cash=initial_cash)
    engine = BacktestEngine(account, market)
    
    # 初始化策略
    strategy = TurtleStrategy(
        entry_period=entry_period,
        exit_period=exit_period,
        atr_period=atr_period,
        risk_per_trade=risk_per_trade,
        stop_loss_atr=stop_loss_atr,
        max_positions=max_positions,
        add_position_atr=add_position_atr
    )
    
    # 获取可用日期
    available_dates = market.get_available_dates(stock_code)
    if not available_dates:
        logger.error(f"未找到股票 {stock_code} 的数据")
        return
    
    logger.info(f"数据范围: {available_dates[0]} 至 {available_dates[-1]}")
    
    def on_trading_day(eng: BacktestEngine, date: str):
        """每个交易日的回调函数"""
        # 所有数据必须从engine获取（带日期保护）
        # 获取当日价格数据
        high = eng.get_high_price(stock_code, date)
        low = eng.get_low_price(stock_code, date)
        close = eng.get_price(stock_code, date)
        
        if high is None or low is None or close is None:
            if len(strategy.price_history) < 5:
                logger.warning(f"日期 {date} 没有价格数据")
            return
        
        # 获取前一日收盘价
        # 需要获取历史数据来计算prev_close
        hist_data = eng.get_historical_data(stock_code, end_date=date)
        if not hist_data.empty and 'prev_close' in hist_data.columns:
            day_data = hist_data[hist_data['datetime'].astype(str).str[:10] == date]
            if not day_data.empty:
                prev_close = float(day_data.iloc[0]['prev_close']) * market.price_adjustment
            else:
                prev_close = close
        else:
            prev_close = close
        
        if high == 0 or low == 0 or close == 0:
            logger.debug(f"日期 {date} 价格数据无效")
            return
        
        # 更新策略价格历史
        strategy.update_price_history(date, high, low, close, prev_close)
        
        # 计算ATR（用于信号和仓位计算）
        if len(strategy.price_history) >= strategy.atr_period:
            df = pd.DataFrame(strategy.price_history)
            atr = strategy.calculate_atr(df, strategy.atr_period)
        else:
            atr = 0.0
        
        # 获取交易信号
        # 海龟策略：买入用最高价判断突破，卖出用最低价判断突破
        # 优先级：卖出 > 买入 > 加仓
        signals = {'buy': False, 'sell': False, 'add_position': False}
        
        # 先检查卖出信号（如果有持仓）
        if strategy.check_sell_signal(low, atr):
            signals['sell'] = True
        # 再检查买入信号（如果无持仓）
        elif strategy.check_buy_signal(high, atr):
            signals['buy'] = True
        # 最后检查加仓信号（如果有持仓且未卖出）
        elif strategy.check_add_position_signal(high, atr):
            signals['add_position'] = True
        
        # 调试：当有足够数据时输出状态
        if len(strategy.price_history) >= strategy.entry_period + 1:
            df = pd.DataFrame(strategy.price_history)
            # 使用前N-1天的数据（不包括当天）
            entry_high = df['high_price'].iloc[-(strategy.entry_period+1):-1].max()
            entry_low = df['low_price'].iloc[-(strategy.entry_period+1):-1].min()
            
            # 计算最高价突破差距（这才是真正用于判断的）
            high_breakthrough_gap = high - entry_high
            
            # 只在接近突破或每20天输出一次
            if (high >= entry_high * 0.98 or len(strategy.price_history) % 20 == 0):
                logger.info(
                    f"[{date}] 收盘={close:.2f}, 最高={high:.2f}, 20日最高={entry_high:.2f}, "
                    f"最高突破差距={high_breakthrough_gap:.2f}, ATR={atr:.2f}, 有持仓={strategy.entry_price is not None}"
                )
            
            # 如果真正突破，输出详细信息
            if high > entry_high:
                logger.warning(
                    f"[{date}] ⚠️ 突破发生！最高={high:.2f} > 20日最高={entry_high:.2f}, "
                    f"差距={high_breakthrough_gap:.2f}"
                )
        
        # 执行卖出信号（优先处理）
        if signals['sell']:
            position = account.get_position(stock_code)
            if position and position['shares'] > 0:
                eng.sell(stock_code, position['shares'])
                logger.info(
                    f"[{date}] 卖出信号: {position['shares']} 股 @ {close:.2f}, "
                    f"止损={strategy.stop_loss:.2f if strategy.stop_loss else 'N/A'}"
                )
        
        # 执行买入信号
        elif signals['buy']:
            # 计算仓位大小
            equity = account.equity({stock_code: close})
            shares = strategy.calculate_position_size(equity, atr, close)
            
            if shares > 0:
                eng.buy(stock_code, shares=shares)
                logger.info(
                    f"[{date}] 买入信号: {shares} 股 @ {close:.2f}, "
                    f"入场价={strategy.entry_price:.2f}, 止损={strategy.stop_loss:.2f}, ATR={atr:.2f}"
                )
        
        # 执行加仓信号
        elif signals['add_position']:
            equity = account.equity({stock_code: close})
            shares = strategy.calculate_position_size(equity, atr, close)
            
            if shares > 0:
                eng.buy(stock_code, shares=shares)
                logger.info(
                    f"[{date}] 加仓信号: {shares} 股 @ {close:.2f}, "
                    f"持仓次数={strategy.position_count}/{strategy.max_positions}, "
                    f"止损={strategy.stop_loss:.2f}"
                )
            position = account.get_position(stock_code)
            if position and position['shares'] > 0:
                eng.sell(stock_code, position['shares'])
                logger.info(
                    f"[{date}] 卖出信号: {position['shares']} 股 @ {close:.2f}"
                )
    
    # 注册交易日回调
    engine.on_date(on_trading_day)
    
    # 运行回测
    engine.run(stock_code, start_date=start_date, end_date=end_date)
    
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
    
    # 输出结果
    logger.info("")
    logger.info("=" * 60)
    logger.info("回测结果")
    logger.info("=" * 60)
    logger.info(f"初始资金: {initial_cash:,.2f} 元")
    logger.info(f"最终权益: {equity:,.2f} 元")
    logger.info(f"总盈亏: {profit:+,.2f} 元")
    logger.info(f"总收益率: {return_pct:+.2f}%")
    logger.info(f"交易次数: {len(account.trades)}")
    logger.info("=" * 60)
    
    # 输出详细账户摘要
    logger.info("")
    logger.info(account.summary(market_prices))


if __name__ == "__main__":
    # 执行海龟策略
    turtle_strategy(
        stock_code="AAPL.O",
        initial_cash=100000.0,
        entry_period=20,
        exit_period=10,
        atr_period=20,
        risk_per_trade=0.02,
        stop_loss_atr=2.0,
        max_positions=4,
        add_position_atr=0.5,
        start_date=None,
        end_date=None
    )

