"""
账户类：用于回测中的账户管理
管理现金、持仓、交易记录等
"""
from datetime import datetime
from typing import Dict, List, Optional
from trader.logger import get_logger

logger = get_logger(__name__)


class Account:
    """模拟交易账户"""
    
    def __init__(self, initial_cash: float = 10000.0):
        """
        初始化账户
        
        Args:
            initial_cash: 初始现金
        """
        self.initial_cash = initial_cash
        self.cash = initial_cash
        self.positions: Dict[str, Dict] = {}  # {stock_code: {"shares": int, "average_price": float}}
        self.trades: List[Dict] = []  # 交易记录
        self.last_update: Optional[datetime] = None
        
    def buy(self, stock_code: str, shares: int, price: float, date: Optional[datetime] = None) -> bool:
        """
        买入股票
        
        Args:
            stock_code: 股票代码
            shares: 买入股数
            price: 买入价格
            date: 交易日期
            
        Returns:
            bool: 是否成功买入
        """
        if shares <= 0:
            logger.warning(f"买入股数必须大于0: {shares}")
            return False
            
        cost = shares * price
        
        if cost > self.cash:
            logger.warning(f"现金不足: 需要 {cost:.2f}, 当前现金 {self.cash:.2f}")
            return False
        
        # 更新现金
        self.cash -= cost
        
        # 更新持仓
        if stock_code in self.positions:
            # 已有持仓，计算新的平均价格
            old_shares = self.positions[stock_code]["shares"]
            old_avg_price = self.positions[stock_code]["average_price"]
            total_cost = old_shares * old_avg_price + cost
            total_shares = old_shares + shares
            new_avg_price = total_cost / total_shares
            
            self.positions[stock_code]["shares"] = total_shares
            self.positions[stock_code]["average_price"] = new_avg_price
        else:
            # 新建持仓
            self.positions[stock_code] = {
                "shares": shares,
                "average_price": price
            }
        
        # 记录交易
        trade = {
            "date": date or datetime.now(),
            "type": "buy",
            "stock_code": stock_code,
            "shares": shares,
            "price": price,
            "cost": cost
        }
        self.trades.append(trade)
        
        self.last_update = date or datetime.now()
        
        logger.info(f"买入 {stock_code}: {shares} 股 @ {price:.2f}, 成本 {cost:.2f}, 剩余现金 {self.cash:.2f}")
        return True
    
    def sell(self, stock_code: str, shares: int, price: float, date: Optional[datetime] = None) -> bool:
        """
        卖出股票
        
        Args:
            stock_code: 股票代码
            shares: 卖出股数
            price: 卖出价格
            date: 交易日期
            
        Returns:
            bool: 是否成功卖出
        """
        if shares <= 0:
            logger.warning(f"卖出股数必须大于0: {shares}")
            return False
        
        if stock_code not in self.positions:
            logger.warning(f"没有持仓: {stock_code}")
            return False
        
        if self.positions[stock_code]["shares"] < shares:
            logger.warning(
                f"持仓不足: 需要卖出 {shares} 股, "
                f"当前持仓 {self.positions[stock_code]['shares']} 股"
            )
            return False
        
        # 计算收益
        revenue = shares * price
        cost = shares * self.positions[stock_code]["average_price"]
        profit = revenue - cost
        
        # 更新现金
        self.cash += revenue
        
        # 更新持仓
        self.positions[stock_code]["shares"] -= shares
        if self.positions[stock_code]["shares"] == 0:
            # 全部卖出，删除持仓
            del self.positions[stock_code]
        
        # 记录交易
        trade = {
            "date": date or datetime.now(),
            "type": "sell",
            "stock_code": stock_code,
            "shares": shares,
            "price": price,
            "revenue": revenue,
            "profit": profit
        }
        self.trades.append(trade)
        
        self.last_update = date or datetime.now()
        
        logger.info(
            f"卖出 {stock_code}: {shares} 股 @ {price:.2f}, "
            f"收入 {revenue:.2f}, 利润 {profit:.2f}, 当前现金 {self.cash:.2f}"
        )
        return True
    
    def equity(self, market_prices: Dict[str, float]) -> float:
        """
        计算账户总权益（现金 + 持仓市值）
        
        Args:
            market_prices: {stock_code: current_price} 当前市场价格字典
            
        Returns:
            float: 总权益
        """
        positions_value = 0.0
        
        for stock_code, position in self.positions.items():
            if stock_code in market_prices:
                current_price = market_prices[stock_code]
                positions_value += position["shares"] * current_price
            else:
                logger.warning(f"无法获取 {stock_code} 的市场价格，使用平均成本价")
                positions_value += position["shares"] * position["average_price"]
        
        return self.cash + positions_value
    
    def get_position(self, stock_code: str) -> Optional[Dict]:
        """
        获取持仓信息
        
        Args:
            stock_code: 股票代码
            
        Returns:
            Optional[Dict]: 持仓信息 {"shares": int, "average_price": float} 或 None
        """
        return self.positions.get(stock_code)
    
    def get_all_positions(self) -> Dict[str, Dict]:
        """
        获取所有持仓
        
        Returns:
            Dict[str, Dict]: 所有持仓信息
        """
        return self.positions.copy()
    
    def get_total_profit(self, market_prices: Dict[str, float]) -> float:
        """
        计算总盈亏
        
        Args:
            market_prices: {stock_code: current_price} 当前市场价格字典
            
        Returns:
            float: 总盈亏（相对于初始资金）
        """
        current_equity = self.equity(market_prices)
        return current_equity - self.initial_cash
    
    def get_total_return(self, market_prices: Dict[str, float]) -> float:
        """
        计算总收益率
        
        Args:
            market_prices: {stock_code: current_price} 当前市场价格字典
            
        Returns:
            float: 总收益率（百分比）
        """
        profit = self.get_total_profit(market_prices)
        return (profit / self.initial_cash) * 100 if self.initial_cash > 0 else 0.0
    
    def summary(self, market_prices: Dict[str, float]) -> str:
        """
        生成账户摘要
        
        Args:
            market_prices: {stock_code: current_price} 当前市场价格字典
            
        Returns:
            str: 账户摘要字符串
        """
        equity = self.equity(market_prices)
        profit = self.get_total_profit(market_prices)
        return_pct = self.get_total_return(market_prices)
        
        lines = [
            "=" * 60,
            "账户摘要",
            "=" * 60,
            f"初始资金:     {self.initial_cash:,.2f} 元",
            f"当前现金:     {self.cash:,.2f} 元",
            f"持仓市值:     {equity - self.cash:,.2f} 元",
            f"总权益:       {equity:,.2f} 元",
            f"总盈亏:       {profit:+,.2f} 元 ({return_pct:+.2f}%)",
            "",
            "持仓明细:",
        ]
        
        if self.positions:
            for stock_code, position in self.positions.items():
                current_price = market_prices.get(stock_code, position["average_price"])
                market_value = position["shares"] * current_price
                profit_per_share = current_price - position["average_price"]
                total_profit = profit_per_share * position["shares"]
                
                lines.append(
                    f"  {stock_code}: "
                    f"{position['shares']} 股 @ 成本 {position['average_price']:.2f}, "
                    f"现价 {current_price:.2f}, "
                    f"市值 {market_value:,.2f}, "
                    f"盈亏 {total_profit:+,.2f}"
                )
        else:
            lines.append("  (无持仓)")
        
        lines.append("=" * 60)
        
        return "\n".join(lines)
