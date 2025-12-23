"""
海龟策略（Turtle Trading Strategy）Agent
经典的趋势跟踪策略，使用突破系统和ATR进行风险管理
"""
from typing import Dict, Optional, List
import pandas as pd
from trader.agent.TradingAgent import TradingAgent
from trader.backtest.engine import BacktestEngine
from trader.backtest.account import Account
from trader.backtest.market import Market
from trader.logger import get_logger

logger = get_logger(__name__)


class TurtleAgent(TradingAgent):
    """
    海龟策略 Agent
    使用突破系统和ATR进行风险管理
    """
    
    def __init__(self, name: str = "TurtleAgent",
                 entry_period: int = 20,  # 突破周期（N日最高/最低）
                 exit_period: int = 10,   # 退出周期
                 atr_period: int = 20,    # ATR计算周期
                 risk_per_trade: float = 0.02,  # 每次交易风险（账户资金的百分比）
                 stop_loss_atr: float = 2.0,   # 止损距离（ATR倍数）
                 max_positions: int = 4,       # 最大加仓次数
                 add_position_atr: float = 0.5):  # 加仓距离（ATR倍数）
        """
        初始化海龟策略 Agent
        
        Args:
            name: Agent 名称
            entry_period: 突破周期，用于计算N日最高/最低价
            exit_period: 退出周期，用于计算退出信号
            atr_period: ATR计算周期
            risk_per_trade: 每次交易的风险（账户资金的百分比）
            stop_loss_atr: 止损距离（ATR的倍数）
            max_positions: 最大加仓次数
            add_position_atr: 加仓距离（ATR的倍数）
        """
        super().__init__(name, max_position_weight=1.0, max_total_weight=1.0)
        self.entry_period = entry_period
        self.exit_period = exit_period
        self.atr_period = atr_period
        self.risk_per_trade = risk_per_trade
        self.stop_loss_atr = stop_loss_atr
        self.max_positions = max_positions
        self.add_position_atr = add_position_atr
        
        # 存储历史数据（按股票代码分别存储）
        self.price_history: Dict[str, List[Dict]] = {}  # {stock_code: [price_data]}
        self.entry_price: Dict[str, Optional[float]] = {}  # {stock_code: entry_price}
        self.stop_loss: Dict[str, Optional[float]] = {}  # {stock_code: stop_loss}
        self.add_position_levels: Dict[str, List[float]] = {}  # {stock_code: [levels]}
        self.position_count: Dict[str, int] = {}  # {stock_code: count}
    
    def _get_or_init_stock_state(self, stock_code: str):
        """获取或初始化股票的状态"""
        if stock_code not in self.price_history:
            self.price_history[stock_code] = []
            self.entry_price[stock_code] = None
            self.stop_loss[stock_code] = None
            self.add_position_levels[stock_code] = []
            self.position_count[stock_code] = 0
    
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
    
    def update_price_history(self, stock_code: str, date: str, high: float, low: float, 
                            close: float, prev_close: float):
        """更新价格历史"""
        self._get_or_init_stock_state(stock_code)
        
        self.price_history[stock_code].append({
            'date': date,
            'high_price': high,
            'low_price': low,
            'close_price': close,
            'prev_close': prev_close
        })
        
        # 只保留必要的历史数据
        max_history = max(self.entry_period, self.exit_period, self.atr_period) + 10
        if len(self.price_history[stock_code]) > max_history:
            self.price_history[stock_code] = self.price_history[stock_code][-max_history:]
    
    def score(self, stock_code: str, engine: BacktestEngine) -> float:
        """
        计算股票的 score（基于海龟策略的信号）
        
        根据突破信号、加仓信号、卖出信号计算看好程度：
        - 买入信号（突破）：返回 1.0
        - 加仓信号：返回 0.5
        - 卖出信号：返回 -1.0
        - 其他情况：返回 0.0
        
        Args:
            stock_code: 股票代码
            engine: 回测引擎
            
        Returns:
            float: score 值，范围 [-1, 1]
        """
        try:
            # 获取当日价格数据
            date = engine.current_date
            if date is None:
                return 0.0
            
            high = engine.get_high_price(stock_code, date)
            low = engine.get_low_price(stock_code, date)
            close = engine.get_price(stock_code, date)
            
            if high is None or low is None or close is None:
                return 0.0
            
            if high == 0 or low == 0 or close == 0:
                return 0.0
            
            # 获取前一日收盘价
            hist_data = engine.get_historical_data(stock_code, end_date=date)
            if not hist_data.empty and 'prev_close' in hist_data.columns:
                day_data = hist_data[hist_data['datetime'].astype(str).str[:10] == date]
                if not day_data.empty:
                    prev_close = float(day_data.iloc[0]['prev_close']) * engine.market.price_adjustment
                else:
                    history = self.price_history.get(stock_code, [])
                    prev_close = history[-1]['close_price'] if history else close
            else:
                history = self.price_history.get(stock_code, [])
                prev_close = history[-1]['close_price'] if history else close
            
            # 更新价格历史
            self.update_price_history(stock_code, date, high, low, close, prev_close)
            
            # 计算ATR
            history = self.price_history.get(stock_code, [])
            if len(history) >= self.atr_period:
                df = pd.DataFrame(history)
                atr = self.calculate_atr(df, self.atr_period)
            else:
                atr = 0.0
            
            # 检查卖出信号（优先级最高）- 使用只读检查方法
            if self._check_sell_signal_readonly(stock_code, low, atr):
                return -1.0  # 卖出信号
            
            # 检查买入信号 - 使用只读检查方法
            if self._check_buy_signal_readonly(stock_code, high, atr):
                return 1.0  # 买入信号
            
            # 检查加仓信号 - 使用只读检查方法
            if self._check_add_position_signal_readonly(stock_code, high, atr):
                return 0.5  # 加仓信号
            
            # 如果已有持仓，返回小的正分（表示继续持有）
            if self.entry_price.get(stock_code) is not None:
                return 0.1
            
            return 0.0
            
        except Exception as e:
            logger.error(f"计算 {stock_code} 的 score 时出错: {e}", exc_info=True)
            return 0.0
    
    def _check_buy_signal_readonly(self, stock_code: str, high_price: float, atr: float) -> bool:
        """检查买入信号（只读，不修改状态）"""
        self._get_or_init_stock_state(stock_code)
        
        history = self.price_history[stock_code]
        if len(history) < self.entry_period:
            return False
        
        if self.entry_price[stock_code] is not None:
            return False  # 已有持仓
        
        df = pd.DataFrame(history)
        # 使用前N-1天的最高价（不包括当天）
        if len(df) >= self.entry_period + 1:
            entry_high = df['high_price'].iloc[-(self.entry_period+1):-1].max()
        else:
            entry_high = df['high_price'].head(-1).tail(self.entry_period).max() if len(df) > self.entry_period else df['high_price'].head(-1).max()
        
        return high_price > entry_high
    
    def check_buy_signal(self, stock_code: str, high_price: float, atr: float) -> bool:
        """检查买入信号（使用最高价）"""
        self._get_or_init_stock_state(stock_code)
        
        history = self.price_history[stock_code]
        if len(history) < self.entry_period:
            return False
        
        if self.entry_price[stock_code] is not None:
            return False  # 已有持仓
        
        df = pd.DataFrame(history)
        # 使用前N-1天的最高价（不包括当天）
        if len(df) >= self.entry_period + 1:
            entry_high = df['high_price'].iloc[-(self.entry_period+1):-1].max()
        else:
            entry_high = df['high_price'].head(-1).tail(self.entry_period).max() if len(df) > self.entry_period else df['high_price'].head(-1).max()
        
        if high_price > entry_high:
            # 触发买入
            self.entry_price[stock_code] = high_price
            self.stop_loss[stock_code] = high_price - (atr * self.stop_loss_atr) if atr > 0 else high_price * 0.95
            self.position_count[stock_code] = 1
            self.add_position_levels[stock_code] = []
            # 计算加仓价格水平
            if atr > 0:
                for i in range(1, self.max_positions):
                    level = high_price + (atr * self.add_position_atr * i)
                    self.add_position_levels[stock_code].append(level)
            return True
        return False
    
    def _check_sell_signal_readonly(self, stock_code: str, low_price: float, atr: float) -> bool:
        """检查卖出信号（只读，不修改状态）"""
        self._get_or_init_stock_state(stock_code)
        
        if self.entry_price[stock_code] is None:
            return False  # 无持仓
        
        df = pd.DataFrame(self.price_history[stock_code])
        # 使用前N-1天的最低价（不包括当天）
        if len(df) >= self.exit_period + 1:
            exit_low = df['low_price'].iloc[-(self.exit_period+1):-1].min()
        else:
            exit_low = df['low_price'].head(-1).tail(self.exit_period).min() if len(df) > self.exit_period else df['low_price'].head(-1).min()
        
        # 卖出信号：跌破退出周期最低价或触发止损
        return low_price < exit_low or (self.stop_loss[stock_code] and low_price < self.stop_loss[stock_code])
    
    def check_sell_signal(self, stock_code: str, low_price: float, atr: float) -> bool:
        """检查卖出信号（使用最低价）"""
        self._get_or_init_stock_state(stock_code)
        
        if self.entry_price[stock_code] is None:
            return False  # 无持仓
        
        df = pd.DataFrame(self.price_history[stock_code])
        # 使用前N-1天的最低价（不包括当天）
        if len(df) >= self.exit_period + 1:
            exit_low = df['low_price'].iloc[-(self.exit_period+1):-1].min()
        else:
            exit_low = df['low_price'].head(-1).tail(self.exit_period).min() if len(df) > self.exit_period else df['low_price'].head(-1).min()
        
        # 卖出信号：跌破退出周期最低价或触发止损
        if low_price < exit_low or (self.stop_loss[stock_code] and low_price < self.stop_loss[stock_code]):
            self.entry_price[stock_code] = None
            self.stop_loss[stock_code] = None
            self.add_position_levels[stock_code] = []
            self.position_count[stock_code] = 0
            return True
        return False
    
    def _check_add_position_signal_readonly(self, stock_code: str, high_price: float, atr: float) -> bool:
        """检查加仓信号（只读，不修改状态）"""
        self._get_or_init_stock_state(stock_code)
        
        if self.entry_price[stock_code] is None:
            return False  # 无持仓
        
        if not self.add_position_levels[stock_code]:
            return False
        
        if self.position_count[stock_code] >= self.max_positions:
            return False
        
        return high_price >= self.add_position_levels[stock_code][0]
    
    def check_add_position_signal(self, stock_code: str, high_price: float, atr: float) -> bool:
        """检查加仓信号"""
        self._get_or_init_stock_state(stock_code)
        
        if self.entry_price[stock_code] is None:
            return False  # 无持仓
        
        if not self.add_position_levels[stock_code]:
            return False
        
        if self.position_count[stock_code] >= self.max_positions:
            return False
        
        if high_price >= self.add_position_levels[stock_code][0]:
            # 触发加仓
            self.position_count[stock_code] += 1
            # 更新止损和加仓水平
            if atr > 0:
                self.stop_loss[stock_code] = high_price - (atr * self.stop_loss_atr)
            self.add_position_levels[stock_code].pop(0)
            return True
        return False
    
    def weight(self, stock_code: str, score: float, engine: BacktestEngine) -> float:
        """
        计算实际资金配置比例（基于ATR和风险控制）
        
        Args:
            stock_code: 股票代码
            score: 该股票的 score 值
            engine: 回测引擎
            
        Returns:
            float: 资金配置比例 [0, max_position_weight]
        """
        # 如果 score 为负（卖出信号），返回 0（不买入）
        if score < 0:
            return 0.0
        
        # 如果 score 为 0，不配置
        if score <= 0:
            return 0.0
        
        try:
            date = engine.current_date
            if date is None:
                return 0.0
            
            # 获取当前价格
            current_price = engine.get_current_price(stock_code)
            if current_price is None or current_price == 0:
                return 0.0
            
            # 获取账户权益（需要所有持仓股票的价格）
            # 获取所有持仓股票代码
            all_positions = list(engine.account.positions.keys())
            if all_positions:
                # 获取所有持仓股票的价格
                market_prices = engine.get_market_prices(all_positions)
            else:
                # 如果没有持仓，只获取当前股票的价格
                market_prices = engine.get_market_prices([stock_code])
            
            account_equity = engine.account.equity(market_prices)
            if account_equity <= 0:
                return 0.0
            
            # 计算ATR
            history = self.price_history.get(stock_code, [])
            if len(history) < self.atr_period:
                return 0.0
            
            df = pd.DataFrame(history)
            atr = self.calculate_atr(df, self.atr_period)
            
            if atr == 0:
                return 0.0
            
            # 基于ATR和风险控制计算仓位大小
            # 风险金额 = 账户权益 * 风险比例
            risk_amount = account_equity * self.risk_per_trade
            
            # 每股风险 = ATR * 止损倍数
            risk_per_share = atr * self.stop_loss_atr
            
            if risk_per_share == 0:
                return 0.0
            
            # 仓位大小（股数）= 风险金额 / 每股风险
            shares = risk_amount / risk_per_share
            
            # 转换为资金配置比例
            position_value = shares * current_price
            weight = position_value / account_equity
            
            # 限制在 [0, max_position_weight] 范围内
            weight = min(weight, self.max_position_weight)
            
            return max(0.0, weight)
            
        except Exception as e:
            logger.error(f"计算 {stock_code} 的 weight 时出错: {e}", exc_info=True)
            return 0.0
    
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
    
    def on_date(self, engine: BacktestEngine, date: str):
        """
        每个交易日的回调函数
        
        实现海龟策略的交易逻辑：使用 score/weight 接口决定交易
        
        Args:
            engine: 回测引擎
            date: 当前日期
        """
        # 获取账户和市场（通过 engine 访问）
        account: Account = engine.account
        
        # 获取所有持仓的股票代码
        stock_codes = list(account.positions.keys())
        
        # 如果没有设置交易股票列表，使用当前持仓的股票
        if not hasattr(self, '_trading_stocks'):
            self._trading_stocks = stock_codes if stock_codes else []
        
        # 如果没有要交易的股票，跳过
        if not self._trading_stocks:
            return
        
        try:
            # 1. 先处理卖出逻辑（检查持仓股票的卖出信号）
            for stock_code in stock_codes:
                if stock_code not in self._trading_stocks:
                    continue
                
                try:
                    # 计算 score（会检查卖出信号）
                    score = self.score(stock_code, engine)
                    
                    # 如果 score < 0，表示卖出信号
                    if score < 0:
                        position = account.get_position(stock_code)
                        if position and position['shares'] > 0:
                            # 获取当前价格
                            close = engine.get_price(stock_code, date)
                            if close:
                                engine.sell(stock_code, position['shares'])
                                stop_loss_value = self.stop_loss.get(stock_code)
                                stop_loss_str = f"{stop_loss_value:.2f}" if stop_loss_value is not None else "N/A"
                                logger.info(
                                    f"[{date}] {stock_code} 卖出信号: {position['shares']} 股 @ {close:.2f}, "
                                    f"止损={stop_loss_str}"
                                )
                except Exception as e:
                    logger.error(f"[{date}] 处理 {stock_code} 卖出信号时出错: {e}", exc_info=True)
            
            # 2. 处理买入逻辑（使用 score/weight 接口）
            # 计算所有交易股票的 score
            scores = self.get_scores(self._trading_stocks, engine)
            
            # 过滤掉负分（卖出信号）和零分
            buy_scores = {code: score for code, score in scores.items() if score > 0}
            
            if not buy_scores:
                return
            
            # 根据 score 决定买入/加仓，并设置状态
            # 需要先调用 check_buy_signal 等方法来设置状态
            for stock_code in buy_scores.keys():
                try:
                    high = engine.get_high_price(stock_code, date)
                    low = engine.get_low_price(stock_code, date)
                    close = engine.get_price(stock_code, date)
                    
                    if high is None or low is None or close is None:
                        continue
                    
                    # 计算ATR
                    history = self.price_history.get(stock_code, [])
                    if len(history) >= self.atr_period:
                        df = pd.DataFrame(history)
                        atr = self.calculate_atr(df, self.atr_period)
                    else:
                        atr = 0.0
                    
                    score = buy_scores[stock_code]
                    
                    # 根据 score 设置状态
                    if score >= 1.0:
                        # 买入信号：调用 check_buy_signal 设置状态
                        if self._check_buy_signal_readonly(stock_code, high, atr):
                            self.check_buy_signal(stock_code, high, atr)  # 设置状态
                    elif score >= 0.5:
                        # 加仓信号：调用 check_add_position_signal 设置状态
                        if self._check_add_position_signal_readonly(stock_code, high, atr):
                            self.check_add_position_signal(stock_code, high, atr)  # 设置状态
                except Exception as e:
                    logger.error(f"[{date}] 设置 {stock_code} 状态时出错: {e}", exc_info=True)
            
            # 计算 weight
            weights = self.get_weights(buy_scores, engine)
            
            # 归一化权重
            normalized_weights = self.normalize_weights(weights)
            
            # 根据 weight 执行买入
            # 获取所有持仓股票的价格（包括当前交易的股票）
            all_positions = list(account.positions.keys())
            all_stocks = list(set(all_positions + self._trading_stocks))
            market_prices = engine.get_market_prices(all_stocks)
            account_equity = engine.account.equity(market_prices)
            
            for stock_code, weight in normalized_weights.items():
                if weight > 0:
                    try:
                        # 获取当前价格
                        close = engine.get_price(stock_code, date)
                        if close is None or close == 0:
                            continue
                        
                        # 计算买入金额
                        amount = account_equity * weight
                        
                        # 获取ATR用于日志
                        history = self.price_history.get(stock_code, [])
                        if len(history) >= self.atr_period:
                            df = pd.DataFrame(history)
                            atr = self.calculate_atr(df, self.atr_period)
                        else:
                            atr = 0.0
                        
                        # 执行买入
                        engine.buy(stock_code, amount=amount)
                        
                        # 记录日志
                        score = buy_scores.get(stock_code, 0.0)
                        stop_loss_value = self.stop_loss.get(stock_code)
                        stop_loss_str = f"{stop_loss_value:.2f}" if stop_loss_value is not None else "N/A"
                        
                        if score >= 1.0:
                            # 买入信号
                            logger.info(
                                f"[{date}] {stock_code} 买入信号: {amount:.2f} 元 @ {close:.2f}, "
                                f"入场价={self.entry_price.get(stock_code, close):.2f}, "
                                f"止损={stop_loss_str}, "
                                f"ATR={atr:.2f}, weight={weight:.4f}"
                            )
                        elif score >= 0.5:
                            # 加仓信号
                            position = account.get_position(stock_code)
                            logger.info(
                                f"[{date}] {stock_code} 加仓信号: {amount:.2f} 元 @ {close:.2f}, "
                                f"总持仓={position['shares'] if position else 0} 股, "
                                f"持仓次数={self.position_count.get(stock_code, 0)}/{self.max_positions}, "
                                f"止损={stop_loss_str}, "
                                f"weight={weight:.4f}"
                            )
                    except Exception as e:
                        logger.error(f"[{date}] 买入 {stock_code} 时出错: {e}", exc_info=True)
                        
        except Exception as e:
            logger.error(f"[{date}] 执行海龟策略时出错: {e}", exc_info=True)
    
    def set_trading_stocks(self, stock_codes: List[str]):
        """
        设置要交易的股票代码列表
        
        Args:
            stock_codes: 股票代码列表
        """
        self._trading_stocks = stock_codes

