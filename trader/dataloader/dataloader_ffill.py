"""
dataloader_ffill: 前向填充数据加载器
使用前向填充（forward fill）补全所有 null 值，包括节假日
"""
from typing import Optional, List
import pandas as pd
from trader.dataloader.dataloader import Dataloader
from trader.dataloader.dataloader_raw import dataloader_raw
from trader.logger import get_logger

logger = get_logger(__name__)


class dataloader_ffill(Dataloader):
    """
    前向填充数据加载器
    使用前向填充（forward fill）补全所有 null 值，包括节假日
    """
    
    def load(self, start_date: str, end_date: str, feature_names: Optional[List[str]] = None, 
             force: bool = False) -> pd.DataFrame:
        """
        加载从开始日期到结束日期的所有 features（前向填充）
        
        Args:
            start_date: 开始日期 (YYYY-MM-DD)
            end_date: 结束日期 (YYYY-MM-DD)
            feature_names: 要加载的特征名称列表，如果为 None 则加载所有特征
            force: 是否强制重新计算，忽略缓存
            
        Returns:
            DataFrame，索引为日期，列为特征名称
            所有 null 值（包括节假日）都会使用前向填充补全
        """
        # 先加载原始数据
        raw_loader = dataloader_raw(self.symbol)
        result = raw_loader.load(start_date, end_date, feature_names, force=force)
        
        # 确保 DataFrame 包含所有日期（包括节假日）
        date_range = self._get_date_range(start_date, end_date)
        if len(result) != len(date_range):
            logger.warning(f"日期数量不匹配: result={len(result)}, date_range={len(date_range)}")
            # 重新索引以确保包含所有日期
            result = result.reindex(date_range)
        
        # 对每个特征列进行前向填充（包括节假日）
        # 先前向填充，如果还有缺失值（例如开始日期是节假日），再后向填充
        for col in result.columns:
            original_nan_count = result[col].isna().sum()
            result[col] = result[col].ffill()
            # 如果还有缺失值，使用后向填充
            if result[col].isna().any():
                result[col] = result[col].bfill()
            final_nan_count = result[col].isna().sum()
            if final_nan_count > 0:
                logger.warning(f"特征 {col} 仍有 {final_nan_count} 个缺失值（原始 {original_nan_count} 个）")
        
        logger.info(f"dataloader_ffill: 加载了 {len(result)} 个日期的数据，"
                   f"所有 null 值（包括节假日）都已使用前向填充补全")
        
        return result

