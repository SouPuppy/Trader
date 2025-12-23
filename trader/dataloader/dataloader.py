"""
Dataloader 基类
提供从开始日期到结束日期的所有 features 加载功能
"""
from abc import ABC, abstractmethod
from typing import Optional, List, Dict
import pandas as pd
import sqlite3
from trader.config import DB_PATH
from trader.logger import get_logger
from trader.features.registry import get_feature_names
from trader.features.cache import get_cached_all_features
from trader.cmd.build_features import compute_feature

logger = get_logger(__name__)


class Dataloader(ABC):
    """
    Dataloader 基类
    可以获取从开始日期到结束日期的所有 features
    """
    
    def __init__(self, symbol: str):
        """
        初始化 Dataloader
        
        Args:
            symbol: 股票代码
        """
        self.symbol = symbol
        self.db_path = DB_PATH
        
        if not self.db_path.exists():
            raise FileNotFoundError(f"数据库文件不存在: {self.db_path}")
    
    def _is_trading_day(self, date: str) -> bool:
        """
        判断某个日期是否是交易日（数据库中是否有数据）
        
        Args:
            date: 日期字符串 (YYYY-MM-DD)
            
        Returns:
            True 如果是交易日，False 如果是节假日
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            query = """
                SELECT COUNT(*) 
                FROM raw_data 
                WHERE stock_code = ? AND datetime = ?
            """
            cursor.execute(query, (self.symbol, date))
            count = cursor.fetchone()[0]
            
            conn.close()
            return count > 0
            
        except Exception as e:
            logger.error(f"判断交易日时出错: {e}", exc_info=True)
            return False
    
    def _get_trading_dates(self, start_date: str, end_date: str) -> List[str]:
        """
        获取日期范围内的所有交易日
        
        Args:
            start_date: 开始日期 (YYYY-MM-DD)
            end_date: 结束日期 (YYYY-MM-DD)
            
        Returns:
            交易日列表
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            query = """
                SELECT DISTINCT datetime 
                FROM raw_data 
                WHERE stock_code = ? 
                  AND datetime >= ? 
                  AND datetime <= ?
                ORDER BY datetime ASC
            """
            cursor.execute(query, (self.symbol, start_date, end_date))
            dates = [row[0] for row in cursor.fetchall()]
            
            conn.close()
            return dates
            
        except Exception as e:
            logger.error(f"获取交易日列表时出错: {e}", exc_info=True)
            return []
    
    def _get_date_range(self, start_date: str, end_date: str) -> pd.DatetimeIndex:
        """
        生成从开始日期到结束日期的所有日期（包括节假日）
        
        Args:
            start_date: 开始日期 (YYYY-MM-DD)
            end_date: 结束日期 (YYYY-MM-DD)
            
        Returns:
            DatetimeIndex，包含所有日期
        """
        start = pd.to_datetime(start_date)
        end = pd.to_datetime(end_date)
        return pd.date_range(start=start, end=end, freq='D')
    
    def _load_features_for_date(self, date: str, force: bool = False) -> Optional[Dict[str, Optional[float]]]:
        """
        加载某个日期的所有特征
        
        Args:
            date: 日期字符串 (YYYY-MM-DD)
            force: 是否强制重新计算，忽略缓存
            
        Returns:
            特征字典 {feature_name: value}，如果是节假日返回 None
        """
        # 检查是否是交易日
        if not self._is_trading_day(date):
            return None
        
        # 获取所有特征名称
        feature_names = get_feature_names()
        
        # 尝试从缓存获取
        if not force:
            cached_features = get_cached_all_features(self.symbol, date)
            if cached_features and any(v is not None for v in cached_features.values()):
                # 检查是否所有特征都已缓存
                missing_features = [name for name in feature_names if name not in cached_features]
                if not missing_features:
                    return cached_features
        
        # 计算所有特征
        features = {}
        for feature_name in feature_names:
            try:
                value = compute_feature(feature_name, date, self.symbol, force=force)
                features[feature_name] = value
            except Exception as e:
                logger.warning(f"计算特征 {feature_name} 时出错: {e}")
                features[feature_name] = None
        
        return features
    
    @abstractmethod
    def load(self, start_date: str, end_date: str, feature_names: Optional[List[str]] = None, 
             force: bool = False) -> pd.DataFrame:
        """
        加载从开始日期到结束日期的所有 features
        
        Args:
            start_date: 开始日期 (YYYY-MM-DD)
            end_date: 结束日期 (YYYY-MM-DD)
            feature_names: 要加载的特征名称列表，如果为 None 则加载所有特征
            force: 是否强制重新计算，忽略缓存
            
        Returns:
            DataFrame，索引为日期，列为特征名称
            节假日返回 None（在对应日期行中）
        """
        pass

