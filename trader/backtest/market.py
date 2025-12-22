"""
市场类：用于回测中的市场数据获取
从数据库读取股票价格数据
"""
import sqlite3
from datetime import datetime
from typing import Dict, List, Optional
import pandas as pd
from trader.config import DB_PATH
from trader.logger import get_logger

logger = get_logger(__name__)


class Market:
    """模拟市场，提供价格数据查询"""
    
    def __init__(self, price_adjustment: float = 1.0):
        """
        初始化市场
        
        Args:
            price_adjustment: 价格调整系数（用于处理价格单位问题，如除以100或1000）
        """
        if not DB_PATH.exists():
            raise FileNotFoundError(f"数据库文件不存在: {DB_PATH}")
        self.db_path = DB_PATH
        self.price_adjustment = price_adjustment
    
    def get_price(self, stock_code: str, date: Optional[str] = None) -> Optional[float]:
        """
        获取股票价格
        
        Args:
            stock_code: 股票代码
            date: 日期（格式: YYYY-MM-DD），如果为 None 则获取最新价格
            
        Returns:
            Optional[float]: 收盘价，如果不存在则返回 None
        """
        try:
            conn = sqlite3.connect(self.db_path)
            
            if date:
                query = """
                    SELECT close_price
                    FROM raw_data
                    WHERE stock_code = ? AND datetime = ?
                    ORDER BY datetime DESC
                    LIMIT 1
                """
                cursor = conn.execute(query, (stock_code, date))
            else:
                query = """
                    SELECT close_price
                    FROM raw_data
                    WHERE stock_code = ?
                    ORDER BY datetime DESC
                    LIMIT 1
                """
                cursor = conn.execute(query, (stock_code,))
            
            row = cursor.fetchone()
            conn.close()
            
            if row and row[0] is not None:
                price = float(row[0]) * self.price_adjustment
                return price
            else:
                logger.warning(f"未找到价格数据: stock_code={stock_code}, date={date}")
                return None
                
        except sqlite3.Error as e:
            logger.error(f"查询价格时出错: {e}")
            return None
    
    def get_price_data(self, stock_code: str, start_date: Optional[str] = None, 
                      end_date: Optional[str] = None) -> pd.DataFrame:
        """
        获取股票历史价格数据
        
        Args:
            stock_code: 股票代码
            start_date: 开始日期（格式: YYYY-MM-DD），可选
            end_date: 结束日期（格式: YYYY-MM-DD），可选
            
        Returns:
            pd.DataFrame: 包含 datetime, close_price 等字段的数据框
        """
        try:
            conn = sqlite3.connect(self.db_path)
            
            query = """
                SELECT 
                    datetime,
                    stock_code,
                    prev_close,
                    open_price,
                    high_price,
                    low_price,
                    close_price,
                    volume
                FROM raw_data
                WHERE stock_code = ?
            """
            params = [stock_code]
            
            if start_date:
                query += " AND datetime >= ?"
                params.append(start_date)
            
            if end_date:
                query += " AND datetime <= ?"
                params.append(end_date)
            
            query += " ORDER BY datetime ASC"
            
            df = pd.read_sql_query(query, conn, params=tuple(params))
            conn.close()
            
            if not df.empty and 'datetime' in df.columns:
                df['datetime'] = pd.to_datetime(df['datetime'])
            
            return df
            
        except sqlite3.Error as e:
            logger.error(f"查询价格数据时出错: {e}")
            return pd.DataFrame()
    
    def get_available_dates(self, stock_code: str) -> List[str]:
        """
        获取股票可用的交易日期列表
        
        Args:
            stock_code: 股票代码
            
        Returns:
            List[str]: 日期列表（格式: YYYY-MM-DD）
        """
        try:
            conn = sqlite3.connect(self.db_path)
            
            query = """
                SELECT DISTINCT datetime
                FROM raw_data
                WHERE stock_code = ?
                ORDER BY datetime ASC
            """
            
            cursor = conn.execute(query, (stock_code,))
            dates = [row[0] for row in cursor.fetchall()]
            conn.close()
            
            return dates
            
        except sqlite3.Error as e:
            logger.error(f"查询可用日期时出错: {e}")
            return []
    
    def get_all_symbols(self) -> List[str]:
        """
        获取数据库中所有股票代码
        
        Returns:
            List[str]: 股票代码列表
        """
        try:
            conn = sqlite3.connect(self.db_path)
            
            query = """
                SELECT DISTINCT stock_code
                FROM raw_data
                ORDER BY stock_code
            """
            
            cursor = conn.execute(query)
            symbols = [row[0] for row in cursor.fetchall()]
            conn.close()
            
            return symbols
            
        except sqlite3.Error as e:
            logger.error(f"查询股票代码时出错: {e}")
            return []
    
    def get_prices_dict(self, stock_codes: List[str], date: Optional[str] = None) -> Dict[str, float]:
        """
        批量获取多个股票的价格
        
        Args:
            stock_codes: 股票代码列表
            date: 日期（格式: YYYY-MM-DD），如果为 None 则获取最新价格
            
        Returns:
            Dict[str, float]: {stock_code: price} 价格字典
        """
        prices = {}
        for stock_code in stock_codes:
            price = self.get_price(stock_code, date)
            if price is not None:
                prices[stock_code] = price
        return prices
