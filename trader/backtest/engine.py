"""
回测引擎：提供行为队列和市场日期循环机制
所有数据访问必须通过engine，带有日期保护
"""
import sys
from datetime import datetime
from typing import List, Dict, Optional, Callable
from collections import deque
from pathlib import Path
import pandas as pd
from trader.backtest.account import Account
from trader.backtest.market import Market
from trader.backtest.report import BacktestReport
from trader.features.registry import get_feature, get_feature_names
from trader.features.cache import (
    get_cached_feature, 
    cache_feature, 
    ensure_features_table,
    get_cached_all_features,
    get_cached_features_batch
)
from trader.dataloader import dataloader_ffill, Dataloader
from trader.logger import get_logger

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False
    # 如果没有 tqdm，创建一个简单的替代品
    def tqdm(iterable, *args, **kwargs):
        return iterable

logger = get_logger(__name__)


class Action:
    """交易行为"""
    
    BUY = "buy"
    SELL = "sell"
    
    def __init__(self, action_type: str, stock_code: str, shares: int = None, 
                 amount: float = None, date: Optional[datetime] = None):
        """
        创建交易行为
        
        Args:
            action_type: 行为类型 ("buy" 或 "sell")
            stock_code: 股票代码
            shares: 股数（可选，与 amount 二选一）
            amount: 金额（可选，买入时使用金额，卖出时使用股数）
            date: 行为日期
        """
        if action_type not in [self.BUY, self.SELL]:
            raise ValueError(f"无效的行为类型: {action_type}")
        
        self.action_type = action_type
        self.stock_code = stock_code
        self.shares = shares
        self.amount = amount
        self.date = date or datetime.now()
    
    def __repr__(self):
        if self.action_type == self.BUY:
            if self.amount:
                return f"Action(BUY {self.stock_code}, amount={self.amount:.2f})"
            else:
                return f"Action(BUY {self.stock_code}, shares={self.shares})"
        else:
            return f"Action(SELL {self.stock_code}, shares={self.shares})"


class BacktestEngine:
    """回测引擎"""
    
    def __init__(self, account: Account, market: Market, enable_report: bool = True, 
                 report_output_dir: Optional[Path] = None, report_title: Optional[str] = None,
                 train_test_split_ratio: float = 0.7, only_test_period: bool = True):
        """
        初始化回测引擎
        
        Args:
            account: 账户实例
            market: 市场实例
            enable_report: 是否启用报告生成
            report_output_dir: 报告输出目录，如果为 None 则使用默认目录
            report_title: 报告标题，用于创建子文件夹和文件名
            train_test_split_ratio: 训练/测试分割比例（默认0.7，即70%用于训练，30%用于测试）
            only_test_period: 是否只运行测试期（默认True，即只运行测试期的交易，训练期只用于模型训练）
        """
        self.account = account
        self.market = market
        self.action_queue: deque = deque()  # 行为队列
        self.current_date: Optional[str] = None
        self.date_index: int = 0
        self.trading_dates: List[str] = []
        self.on_date_callbacks: List[Callable] = []  # 每个交易日回调函数列表
        self.enable_report = enable_report
        self.train_test_split_ratio = train_test_split_ratio
        self.train_test_split_date: Optional[str] = None  # 训练/测试分割日期（在 run 时计算）
        self.only_test_period = only_test_period  # 是否只运行测试期
        self.report_title = report_title  # 保存 report_title，用于判断是否生成报告文件
        self.report: Optional[BacktestReport] = None
        if enable_report:
            self.report = BacktestReport(report_output_dir, title=report_title)
    
    def add_action(self, action: Action):
        """
        添加行为到队列
        
        Args:
            action: 交易行为
        """
        self.action_queue.append(action)
        logger.debug(f"添加行为到队列: {action}")
    
    def buy(self, stock_code: str, shares: int = None, amount: float = None):
        """
        提交买入订单
        
        Args:
            stock_code: 股票代码
            shares: 买入股数（可选）
            amount: 买入金额（可选，与 shares 二选一）
        """
        # 检查是否在训练期（训练期不允许交易）
        if self.train_test_split_date and self.current_date:
            if self.current_date < self.train_test_split_date:
                logger.debug(f"训练期禁止交易，跳过买入订单: {stock_code}")
                return
        
        action = Action(Action.BUY, stock_code, shares=shares, amount=amount, date=self.current_date)
        self.add_action(action)
    
    def sell(self, stock_code: str, shares: int):
        """
        提交卖出订单
        
        Args:
            stock_code: 股票代码
            shares: 卖出股数
        """
        # 检查是否在训练期（训练期不允许交易）
        if self.train_test_split_date and self.current_date:
            if self.current_date < self.train_test_split_date:
                logger.debug(f"训练期禁止交易，跳过卖出订单: {stock_code}")
                return
        
        action = Action(Action.SELL, stock_code, shares=shares, date=self.current_date)
        self.add_action(action)
    
    def on_date(self, callback: Callable):
        """
        注册每个交易日的回调函数
        
        Args:
            callback: 回调函数，接收 (engine, date) 参数
        """
        self.on_date_callbacks.append(callback)
    
    def _execute_actions(self):
        """执行队列中的所有行为"""
        # 检查是否在训练期（如果设置了分割日期，训练期不允许交易）
        if self.train_test_split_date and self.current_date:
            if self.current_date < self.train_test_split_date:
                # 训练期，清空队列，不允许交易
                while self.action_queue:
                    action = self.action_queue.popleft()
                    logger.debug(f"训练期禁止交易，跳过行为: {action}")
                return
        
        while self.action_queue:
            action = self.action_queue.popleft()
            
            # 获取当前价格
            price = self.market.get_price(action.stock_code, self.current_date)
            if price is None:
                logger.warning(f"无法获取 {action.stock_code} 在 {self.current_date} 的价格，跳过行为: {action}")
                continue
            
            # 执行买入
            if action.action_type == Action.BUY:
                if action.amount:
                    # 按金额买入（允许买入小数股，实际按金额计算）
                    # 如果价格太高买不到1股，仍然买入，使用实际金额
                    cost = min(action.amount, self.account.cash)  # 不超过可用现金
                    shares = int(cost / price)  # 向下取整
                    
                    if shares > 0:
                        self.account.buy(action.stock_code, shares, price, 
                                       datetime.strptime(self.current_date, "%Y-%m-%d"))
                    else:
                        # 价格太高，买不到1股
                        logger.warning(f"金额 {action.amount:.2f} 无法买入至少1股 (价格: {price:.2f})，跳过")
                elif action.shares:
                    # 按股数买入
                    cost = action.shares * price
                    if cost <= self.account.cash:
                        self.account.buy(action.stock_code, action.shares, price,
                                       datetime.strptime(self.current_date, "%Y-%m-%d"))
                    else:
                        logger.warning(f"现金不足，无法买入 {action.shares} 股 {action.stock_code}")
            
            # 执行卖出
            elif action.action_type == Action.SELL:
                if action.shares:
                    self.account.sell(action.stock_code, action.shares, price,
                                    datetime.strptime(self.current_date, "%Y-%m-%d"))
    
    def run(self, stock_code: str, start_date: Optional[str] = None, 
            end_date: Optional[str] = None):
        """
        运行回测
        
        Args:
            stock_code: 股票代码
            start_date: 开始日期（格式: YYYY-MM-DD），如果为 None 则从最早可用日期开始
            end_date: 结束日期（格式: YYYY-MM-DD），如果为 None 则到最新日期
        """
        # 获取可用日期
        available_dates = self.market.get_available_dates(stock_code)
        if not available_dates:
            logger.error(f"未找到股票 {stock_code} 的数据")
            return
        
        # 确定开始和结束日期
        if start_date is None:
            start_date = available_dates[0]
        if end_date is None:
            end_date = available_dates[-1]
        
        # 过滤日期范围
        self.trading_dates = [d for d in available_dates if start_date <= d <= end_date]
        
        if not self.trading_dates:
            logger.error(f"在指定日期范围内没有交易数据")
            return
        
        # 计算训练/测试分割日期
        if self.train_test_split_ratio > 0 and self.train_test_split_ratio < 1:
            split_index = int(len(self.trading_dates) * self.train_test_split_ratio)
            if split_index > 0 and split_index < len(self.trading_dates):
                self.train_test_split_date = self.trading_dates[split_index]
                # 设置到报告中
                if self.report:
                    self.report.set_train_test_split_date(self.train_test_split_date)
                logger.info(
                    f"训练/测试分割: {self.train_test_split_ratio*100:.0f}% / {(1-self.train_test_split_ratio)*100:.0f}%, "
                    f"分割日期: {self.train_test_split_date}"
                )
        
        # 确定要遍历的日期
        # 如果 only_test_period=True，仍然需要遍历训练期的日期（用于模型训练），但不执行交易
        dates_to_iterate = self.trading_dates
        
        # 确定测试期的日期（用于执行交易和记录报告）
        test_period_dates = dates_to_iterate
        if self.only_test_period and self.train_test_split_date:
            test_period_dates = [d for d in self.trading_dates if d >= self.train_test_split_date]
            train_dates_count = len(self.trading_dates) - len(test_period_dates)
            logger.info(
                f"训练/测试模式: 训练期 {train_dates_count} 个交易日（仅用于模型训练，不执行交易）, "
                f"测试期 {len(test_period_dates)} 个交易日（执行交易和记录）"
            )
        else:
            logger.info(f"完整回测模式: 运行全部 {len(dates_to_iterate)} 个交易日（训练期和测试期都执行交易）")
        
        if not dates_to_iterate:
            logger.warning("没有可遍历的交易日")
            return
        
        test_start_date = test_period_dates[0] if test_period_dates else dates_to_iterate[0]
        test_end_date = test_period_dates[-1] if test_period_dates else dates_to_iterate[-1]
        logger.info(f"开始回测: {dates_to_iterate[0]} 至 {dates_to_iterate[-1]}, 共 {len(dates_to_iterate)} 个交易日")
        if self.only_test_period and test_period_dates:
            logger.info(f"测试期范围: {test_start_date} 至 {test_end_date}, 共 {len(test_period_dates)} 个交易日")
        
        # 按日期循环（遍历所有日期，但根据 only_test_period 决定是否执行交易）
        # 使用 tqdm 显示进度条（固定在底部，不干扰日志）
        date_iterator = tqdm(
            enumerate(dates_to_iterate),
            total=len(dates_to_iterate),
            desc="回测进度",
            unit="日",
            disable=not HAS_TQDM,
            file=sys.stderr,  # 明确输出到 stderr
            position=0,  # 主进度条固定在底部（position=0）
            leave=True,  # 完成后保留进度条
            ncols=100  # 限制宽度，避免太宽
        )
        for self.date_index, date in date_iterator:
            self.current_date = date
            
            # 判断当前是否在测试期
            is_test_period = True
            if self.train_test_split_date:
                is_test_period = date >= self.train_test_split_date
            
            # 调用每个交易日的回调函数（训练期和测试期都调用，用于模型训练）
            for callback in self.on_date_callbacks:
                try:
                    callback(self, date)
                except Exception as e:
                    logger.error(f"执行回调函数时出错: {e}", exc_info=True)
            
            # 决定是否执行交易：
            # - 如果 only_test_period=True：只在测试期执行交易
            # - 如果 only_test_period=False：训练期和测试期都执行交易
            should_execute_trades = True
            if self.only_test_period:
                should_execute_trades = is_test_period
            
            if should_execute_trades:
                # 执行队列中的行为
                self._execute_actions()
            
            # 只在测试期记录每日账户状态（用于生成报告）
            if self.enable_report and self.report and is_test_period:
                # 获取当前所有持仓的市场价格
                market_prices = {}
                for stock_code in self.account.positions.keys():
                    price = self.market.get_price(stock_code, date)
                    if price is not None:
                        market_prices[stock_code] = price
                
                # 同时获取所有交易股票的价格（用于多资产回测）
                # 如果账户中有多个持仓，尝试获取所有相关股票的价格
                if len(self.account.positions) > 1:
                    # 对于多资产回测，确保获取所有相关股票的价格
                    all_traded_stocks = set(self.account.positions.keys())
                    # 从交易记录中提取所有交易过的股票
                    for trade in self.account.trades:
                        all_traded_stocks.add(trade['stock_code'])
                    
                    for code in all_traded_stocks:
                        if code not in market_prices:
                            price = self.market.get_price(code, date)
                            if price is not None:
                                market_prices[code] = price
                
                # 记录每日状态
                self.report.record_daily_state(date, self.account, market_prices)
        
        logger.info(f"回测完成: {len(self.trading_dates)} 个交易日")
        
        # 生成报告（只有当 report_title 不为 None 时才生成报告文件）
        # 如果 report_title 为 None，只记录 daily_records，不生成报告文件
        if self.enable_report and self.report and self.report_title is not None:
            try:
                # 提取所有交易过的股票代码（用于多资产回测）
                all_traded_stocks = set()
                all_traded_stocks.add(stock_code)  # 添加主要股票代码
                for trade in self.account.trades:
                    all_traded_stocks.add(trade['stock_code'])
                all_traded_stocks = sorted(list(all_traded_stocks))
                
                report_file = self.report.generate_report(
                    self.account, stock_code, start_date, end_date, 
                    all_stock_codes=all_traded_stocks
                )
                logger.info(f"回测报告已生成: {report_file}")
            except Exception as e:
                logger.error(f"生成报告时出错: {e}", exc_info=True)
    
    def is_in_test_period(self, date: Optional[str] = None) -> bool:
        """
        判断指定日期是否在测试期
        
        Args:
            date: 日期（格式: YYYY-MM-DD），如果为 None 则使用当前日期
            
        Returns:
            bool: 是否在测试期
        """
        if not self.train_test_split_date:
            return False
        
        check_date = date if date else self.current_date
        if not check_date:
            return False
        
        return check_date >= self.train_test_split_date
    
    def get_train_test_split_date(self) -> Optional[str]:
        """
        获取训练/测试分割日期
        
        Returns:
            Optional[str]: 分割日期（格式: YYYY-MM-DD），如果未设置则返回 None
        """
        return self.train_test_split_date
    
    def get_current_price(self, stock_code: str) -> Optional[float]:
        """
        获取当前日期的价格（带日期保护）
        
        Args:
            stock_code: 股票代码
            
        Returns:
            Optional[float]: 当前价格
        """
        return self.get_price(stock_code, date=None)
    
    def get_market_prices(self, stock_codes: List[str]) -> Dict[str, float]:
        """
        获取多个股票的当前价格
        
        Args:
            stock_codes: 股票代码列表
            
        Returns:
            Dict[str, float]: {stock_code: price} 价格字典
        """
        return self.market.get_prices_dict(stock_codes, self.current_date)
    
    def _check_date_access(self, date: Optional[str] = None) -> str:
        """
        检查日期访问权限（日期保护）
        
        注意：允许访问训练期的历史数据（只要不超过当前日期）。
        这样可以确保在测试期开始时，可以使用训练期的数据来计算滑动窗口特征。
        
        Args:
            date: 要访问的日期，如果为 None 则使用当前日期
            
        Returns:
            str: 允许访问的日期
            
        Raises:
            ValueError: 如果尝试访问未来数据
        """
        if self.current_date is None:
            raise ValueError("回测尚未开始，无法访问数据")
        
        access_date = date or self.current_date
        
        # 检查是否尝试访问未来数据
        # 注意：允许访问训练期的历史数据（只要不超过当前日期）
        if access_date > self.current_date:
            raise ValueError(
                f"不允许访问未来数据: 当前日期={self.current_date}, "
                f"请求日期={access_date}"
            )
        
        return access_date
    
    def get_price_data(self, stock_code: str, date: Optional[str] = None, 
                      lookback: int = 0) -> pd.DataFrame:
        """
        获取股票价格数据（带日期保护）
        
        注意：在测试期，可以访问训练期的历史数据用于计算滑动窗口特征。
        这样可以避免在测试期开始时因为数据不足而无法计算特征。
        
        Args:
            stock_code: 股票代码
            date: 日期（格式: YYYY-MM-DD），如果为 None 则使用当前日期
            lookback: 回看窗口大小（需要多少天的历史数据，已废弃，保留用于兼容性）
            
        Returns:
            pd.DataFrame: 价格数据，包含从最早可用日期到指定日期的所有数据
                         （包括训练期的数据，如果当前在测试期）
            
        Raises:
            ValueError: 如果尝试访问未来数据
        """
        access_date = self._check_date_access(date)
        # 获取所有到 access_date 的数据（包括训练期的数据）
        # 这样在测试期开始时，可以使用训练期的数据来计算滑动窗口特征
        return self.market.get_price_data(stock_code, end_date=access_date)
    
    def get_price(self, stock_code: str, date: Optional[str] = None) -> Optional[float]:
        """
        获取股票价格（带日期保护）
        
        Args:
            stock_code: 股票代码
            date: 日期（格式: YYYY-MM-DD），如果为 None 则使用当前日期
            
        Returns:
            Optional[float]: 价格
            
        Raises:
            ValueError: 如果尝试访问未来数据
        """
        access_date = self._check_date_access(date)
        return self.market.get_price(stock_code, access_date)
    
    def get_high_price(self, stock_code: str, date: Optional[str] = None) -> Optional[float]:
        """获取最高价（带日期保护）"""
        access_date = self._check_date_access(date)
        df = self.market.get_price_data(stock_code, end_date=access_date)
        if df.empty:
            return None
        day_data = df[df['datetime'].astype(str).str[:10] == access_date]
        if day_data.empty:
            return None
        return float(day_data.iloc[0]['high_price']) * self.market.price_adjustment
    
    def get_low_price(self, stock_code: str, date: Optional[str] = None) -> Optional[float]:
        """获取最低价（带日期保护）"""
        access_date = self._check_date_access(date)
        df = self.market.get_price_data(stock_code, end_date=access_date)
        if df.empty:
            return None
        day_data = df[df['datetime'].astype(str).str[:10] == access_date]
        if day_data.empty:
            return None
        return float(day_data.iloc[0]['low_price']) * self.market.price_adjustment
    
    def get_open_price(self, stock_code: str, date: Optional[str] = None) -> Optional[float]:
        """获取开盘价（带日期保护）"""
        access_date = self._check_date_access(date)
        df = self.market.get_price_data(stock_code, end_date=access_date)
        if df.empty:
            return None
        day_data = df[df['datetime'].astype(str).str[:10] == access_date]
        if day_data.empty:
            return None
        return float(day_data.iloc[0]['open_price']) * self.market.price_adjustment
    
    def get_volume(self, stock_code: str, date: Optional[str] = None) -> Optional[float]:
        """获取成交量（带日期保护）"""
        access_date = self._check_date_access(date)
        df = self.market.get_price_data(stock_code, end_date=access_date)
        if df.empty:
            return None
        day_data = df[df['datetime'].astype(str).str[:10] == access_date]
        if day_data.empty:
            return None
        return float(day_data.iloc[0]['volume'])
    
    def get_feature(self, feature_name: str, stock_code: str, 
                   date: Optional[str] = None, force: bool = False) -> Optional[float]:
        """
        获取特征值（带日期保护）
        
        Args:
            feature_name: 特征名称
            stock_code: 股票代码
            date: 日期（格式: YYYY-MM-DD），如果为 None 则使用当前日期
            force: 是否强制重新计算，忽略缓存
            
        Returns:
            Optional[float]: 特征值
            
        Raises:
            ValueError: 如果尝试访问未来数据或特征不存在
        """
        access_date = self._check_date_access(date)
        
        # 检查缓存（如果 force=False）
        if not force:
            cached_value = get_cached_feature(stock_code, access_date, feature_name)
            if cached_value is not None:
                logger.debug(f"从缓存获取特征: {feature_name} for {stock_code} on {access_date} = {cached_value}")
                return cached_value
        
        # 获取特征规范
        feature_spec = get_feature(feature_name)
        if feature_spec is None:
            raise ValueError(f"特征不存在: {feature_name}")
        
        # 加载所需的历史数据
        # 注意：在测试期，这会包含训练期的数据，可以用于计算滑动窗口特征
        # 例如：如果当前是测试期的第一天，但需要20日均线，可以使用训练期的数据
        df = self.market.get_price_data(stock_code, end_date=access_date)
        if df.empty:
            return None
        
        # 应用价格调整
        for col in ['prev_close', 'open_price', 'high_price', 'low_price', 'close_price']:
            if col in df.columns:
                df[col] = df[col] * self.market.price_adjustment
        
        # 确保datetime是索引
        if 'datetime' in df.columns:
            df['datetime'] = pd.to_datetime(df['datetime'])
            df = df.sort_values('datetime')
            df = df.set_index('datetime')
        
        # 只保留到访问日期的数据
        access_date_dt = pd.to_datetime(access_date)
        df = df[df.index <= access_date_dt]
        
        # 如果数据不足，返回None
        if len(df) < feature_spec.lookback + 1:
            logger.debug(f"数据不足: {stock_code} on {access_date}, feature={feature_name}, "
                        f"需要 {feature_spec.lookback + 1} 行, 实际 {len(df)} 行")
            return None
        
        # 计算特征
        try:
            logger.debug(f"计算特征: {feature_name} for {stock_code} on {access_date}")
            result_series = feature_spec.compute(df)
            if result_series.empty:
                logger.debug(f"特征计算结果为空: {feature_name} for {stock_code} on {access_date}")
                return None
            
            # 返回指定日期的值
            if access_date_dt in result_series.index:
                value = result_series.loc[access_date_dt]
            else:
                # 如果没有精确匹配，返回最新值（不超过访问日期）
                available_values = result_series[result_series.index <= access_date_dt]
                if available_values.empty:
                    logger.debug(f"没有可用的特征值: {feature_name} for {stock_code} on {access_date}")
                    return None
                value = available_values.iloc[-1]
            
            result = float(value) if pd.notna(value) else None
            logger.debug(f"特征计算结果: {feature_name} for {stock_code} on {access_date} = {result}")
            
            # 存储到缓存
            cache_feature(stock_code, access_date, feature_name, result)
            
            return result
        except Exception as e:
            logger.error(f"计算特征 {feature_name} 时出错: {e}", exc_info=True)
            return None
    
    def get_features(self, feature_names: List[str], stock_code: str,
                    date: Optional[str] = None, force: bool = False) -> Dict[str, Optional[float]]:
        """
        批量获取多个特征值（带日期保护）
        使用 wide format 缓存，一次查询获取多个特征
        
        Args:
            feature_names: 特征名称列表
            stock_code: 股票代码
            date: 日期（格式: YYYY-MM-DD），如果为 None 则使用当前日期
            force: 是否强制重新计算，忽略缓存
            
        Returns:
            Dict[str, Optional[float]]: {feature_name: value} 特征值字典
        """
        access_date = self._check_date_access(date)
        
        # 确保 features 表存在
        ensure_features_table()
        
        logger.debug(f"批量获取特征: {stock_code} on {access_date}, 共 {len(feature_names)} 个特征, force={force}")
        
        # 检查缓存（如果 force=False）
        result = {}
        if not force:
            cached_data = get_cached_features_batch(stock_code, [access_date], feature_names)
            if access_date in cached_data:
                cached_features = cached_data[access_date]
                # 检查是否有缺失的特征
                missing_features = set(feature_names) - set(cached_features.keys())
                if not missing_features:
                    logger.debug(f"从缓存批量获取特征: {stock_code} on {access_date}, 共 {len(cached_features)} 个特征")
                    return cached_features
                else:
                    # 部分特征在缓存中，部分需要计算
                    result.update(cached_features)
                    feature_names = list(missing_features)
        
        # 计算缺失的特征
        try:
            from tqdm import tqdm
        except ImportError:
            def tqdm(iterable, desc=None, total=None, **kwargs):
                if desc:
                    logger.debug(f"{desc}...")
                return iterable
        
        for feature_name in tqdm(
            feature_names, 
            desc=f"获取特征 ({stock_code})", 
            leave=False,  # 完成后清除，不占用位置
            unit="特征",
            file=sys.stderr,  # 明确输出到 stderr
            position=1,  # 嵌套进度条，使用 position=1（在主进度条上方）
            ncols=100  # 限制宽度，避免太宽
        ):
            result[feature_name] = self.get_feature(feature_name, stock_code, date, force=force)
        
        logger.debug(f"特征获取完成: {stock_code} on {access_date}, 共 {len(result)} 个特征")
        return result
    
    def get_all_features(self, stock_code: str, 
                        date: Optional[str] = None, force: bool = False) -> Dict[str, Optional[float]]:
        """
        获取所有特征值（带日期保护）
        使用 wide format 缓存，一次查询获取所有特征
        
        Args:
            stock_code: 股票代码
            date: 日期（格式: YYYY-MM-DD），如果为 None 则使用当前日期
            force: 是否强制重新计算，忽略缓存
            
        Returns:
            Dict[str, Optional[float]]: {feature_name: value} 所有特征值字典
        """
        access_date = self._check_date_access(date)
        
        # 检查缓存（如果 force=False）
        if not force:
            cached_features = get_cached_all_features(stock_code, access_date)
            if cached_features:
                # 检查是否有缺失的特征
                feature_names = get_feature_names()
                missing_features = set(feature_names) - set(cached_features.keys())
                if not missing_features:
                    logger.debug(f"从缓存获取所有特征: {stock_code} on {access_date}, 共 {len(cached_features)} 个特征")
                    return cached_features
        
        # 如果缓存不完整或 force=True，计算所有特征
        feature_names = get_feature_names()
        result = {}
        for feature_name in feature_names:
            result[feature_name] = self.get_feature(feature_name, stock_code, date, force=force)
        
        return result
    
    def get_historical_data(self, stock_code: str, start_date: Optional[str] = None,
                           end_date: Optional[str] = None) -> pd.DataFrame:
        """
        获取历史数据（带日期保护）
        
        Args:
            stock_code: 股票代码
            start_date: 开始日期（格式: YYYY-MM-DD），如果为 None 则从最早开始
            end_date: 结束日期（格式: YYYY-MM-DD），如果为 None 则使用当前日期
            
        Returns:
            pd.DataFrame: 历史数据
            
        Raises:
            ValueError: 如果尝试访问未来数据
        """
        access_end_date = self._check_date_access(end_date)
        return self.market.get_price_data(stock_code, start_date=start_date, 
                                         end_date=access_end_date)
    
    def get_features_with_dataloader(self, stock_code: str, 
                                    start_date: Optional[str] = None,
                                    end_date: Optional[str] = None,
                                    feature_names: Optional[List[str]] = None,
                                    dataloader: Optional[Dataloader] = None,
                                    force: bool = False) -> pd.DataFrame:
        """
        使用 dataloader 获取特征数据（带日期保护和数据补全）
        
        该方法使用 dataloader 加载从开始日期到结束日期的所有特征，支持数据补全。
        默认使用 ffill（前向填充）补全缺失数据。
        
        Args:
            stock_code: 股票代码
            start_date: 开始日期（格式: YYYY-MM-DD），如果为 None 则从最早开始
            end_date: 结束日期（格式: YYYY-MM-DD），如果为 None 则使用当前日期
            feature_names: 要加载的特征名称列表，如果为 None 则加载所有特征
            dataloader: 数据加载器实例，如果为 None 则使用默认的 dataloader_ffill
            force: 是否强制重新计算，忽略缓存
            
        Returns:
            pd.DataFrame: 特征数据，索引为日期，列为特征名称
            所有缺失值（包括节假日）都会根据 dataloader 的策略进行补全
            
        Raises:
            ValueError: 如果尝试访问未来数据
        """
        access_end_date = self._check_date_access(end_date)
        
        # 如果没有指定开始日期，从最早可用日期开始
        if start_date is None:
            available_dates = self.market.get_available_dates(stock_code)
            if not available_dates:
                logger.warning(f"未找到股票 {stock_code} 的数据")
                return pd.DataFrame()
            start_date = available_dates[0]
        
        # 使用默认的 ffill dataloader（如果未指定）
        if dataloader is None:
            dataloader = dataloader_ffill(stock_code)
        
        # 使用 dataloader 加载数据
        logger.debug(f"使用 dataloader 加载特征: {stock_code} from {start_date} to {access_end_date}")
        result = dataloader.load(start_date, access_end_date, feature_names, force=force)
        
        # 确保日期索引是 datetime 类型
        if not result.empty and not isinstance(result.index, pd.DatetimeIndex):
            try:
                result.index = pd.to_datetime(result.index)
            except Exception as e:
                logger.warning(f"无法将索引转换为 DatetimeIndex: {e}")
        
        logger.debug(f"dataloader 加载完成: {stock_code}, 共 {len(result)} 行, {len(result.columns)} 列")
        return result
    
    def get_features_for_date_with_dataloader(self, stock_code: str,
                                             date: Optional[str] = None,
                                             feature_names: Optional[List[str]] = None,
                                             dataloader: Optional[Dataloader] = None,
                                             force: bool = False) -> Dict[str, Optional[float]]:
        """
        使用 dataloader 获取指定日期的特征值（带日期保护和数据补全）
        
        这是一个便捷方法，用于获取单个日期的所有特征值。
        内部使用 get_features_with_dataloader 加载数据，然后返回指定日期的行。
        
        Args:
            stock_code: 股票代码
            date: 日期（格式: YYYY-MM-DD），如果为 None 则使用当前日期
            feature_names: 要加载的特征名称列表，如果为 None 则加载所有特征
            dataloader: 数据加载器实例，如果为 None 则使用默认的 dataloader_ffill
            force: 是否强制重新计算，忽略缓存
            
        Returns:
            Dict[str, Optional[float]]: {feature_name: value} 特征值字典
            
        Raises:
            ValueError: 如果尝试访问未来数据
        """
        access_date = self._check_date_access(date)
        
        # 使用 dataloader 加载数据（从当前日期往前加载一些历史数据，以确保补全）
        # 这里我们加载从当前日期往前 30 天的数据，以确保有足够的历史数据用于补全
        from datetime import datetime, timedelta
        try:
            date_obj = datetime.strptime(access_date, "%Y-%m-%d")
            start_date_obj = date_obj - timedelta(days=30)
            start_date = start_date_obj.strftime("%Y-%m-%d")
        except ValueError:
            start_date = access_date
        
        df = self.get_features_with_dataloader(
            stock_code, 
            start_date=start_date,
            end_date=access_date,
            feature_names=feature_names,
            dataloader=dataloader,
            force=force
        )
        
        if df.empty:
            logger.warning(f"未找到 {stock_code} 在 {access_date} 的特征数据")
            return {}
        
        # 获取指定日期的数据
        try:
            date_dt = pd.to_datetime(access_date)
            if date_dt in df.index:
                row = df.loc[date_dt]
            else:
                # 如果没有精确匹配，返回最新值（不超过访问日期）
                available_rows = df[df.index <= date_dt]
                if available_rows.empty:
                    logger.warning(f"未找到 {stock_code} 在 {access_date} 或之前的数据")
                    return {}
                row = available_rows.iloc[-1]
            
            # 转换为字典
            result = {}
            for col in df.columns:
                value = row[col]
                result[col] = float(value) if pd.notna(value) else None
            
            return result
            
        except Exception as e:
            logger.error(f"获取 {stock_code} 在 {access_date} 的特征数据时出错: {e}", exc_info=True)
            return {}

