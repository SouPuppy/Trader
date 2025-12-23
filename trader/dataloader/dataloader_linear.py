"""
dataloader_linear: 线性插值数据加载器
使用线性插值补全所有 null 值，包括节假日
"""
from typing import Optional, List
import pandas as pd
from trader.dataloader.dataloader import Dataloader
from trader.dataloader.dataloader_raw import dataloader_raw
from trader.logger import get_logger

logger = get_logger(__name__)


class dataloader_linear(Dataloader):
    """
    线性插值数据加载器
    使用线性插值补全所有 null 值，包括节假日
    """
    
    def load(self, start_date: str, end_date: str, feature_names: Optional[List[str]] = None, 
             force: bool = False) -> pd.DataFrame:
        """
        加载从开始日期到结束日期的所有 features（线性插值）
        
        Args:
            start_date: 开始日期 (YYYY-MM-DD)
            end_date: 结束日期 (YYYY-MM-DD)
            feature_names: 要加载的特征名称列表，如果为 None 则加载所有特征
            force: 是否强制重新计算，忽略缓存
            
        Returns:
            DataFrame，索引为日期，列为特征名称
            所有 null 值（包括节假日）都会使用线性插值补全
        """
        # 先加载原始数据
        raw_loader = dataloader_raw(self.symbol)
        result = raw_loader.load(start_date, end_date, feature_names, force=force)
        
        # 确保 DataFrame 包含所有日期（包括节假日）
        date_range = self._get_date_range(start_date, end_date)
        # 强制重新索引以确保索引完全匹配
        result = result.reindex(date_range)
        
        # 对每个特征列进行线性插值（包括节假日）
        # 先线性插值，如果还有缺失值（例如开始或结束日期是节假日），再后向/前向填充
        for col in result.columns:
            original_nan_count = result[col].isna().sum()
            result[col] = result[col].interpolate(method='linear', limit_direction='both')
            # 如果还有缺失值，使用前向和后向填充
            if result[col].isna().any():
                result[col] = result[col].ffill()
                result[col] = result[col].bfill()
            # 如果还有缺失值（例如整个列都是 NaN），填充为 0 以确保线条连续
            final_nan_count = result[col].isna().sum()
            if final_nan_count > 0:
                logger.warning(f"特征 {col} 仍有 {final_nan_count} 个缺失值（原始 {original_nan_count} 个），填充为 0")
                result[col] = result[col].fillna(0)
        
        logger.info(f"dataloader_linear: 加载了 {len(result)} 个日期的数据，"
                   f"所有 null 值（包括节假日）都已使用线性插值补全")
        
        return result

