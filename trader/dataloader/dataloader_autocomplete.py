"""
dataloader_autocomplete: 自动插值补全数据加载器
自动插值补全所有 null 值，包括节假日（使用前后交易日的数据）
"""
from typing import Optional, List
import pandas as pd
import numpy as np
from trader.dataloader.dataloader import Dataloader
from trader.dataloader.dataloader_raw import dataloader_raw
from trader.logger import get_logger

logger = get_logger(__name__)


class dataloader_autocomplete(Dataloader):
    """
    自动插值补全数据加载器
    自动插值补全所有 null 值，包括节假日（使用前后交易日的数据）
    """
    
    def load(self, start_date: str, end_date: str, feature_names: Optional[List[str]] = None, 
             force: bool = False) -> pd.DataFrame:
        """
        加载从开始日期到结束日期的所有 features（自动插值补全）
        
        Args:
            start_date: 开始日期 (YYYY-MM-DD)
            end_date: 结束日期 (YYYY-MM-DD)
            feature_names: 要加载的特征名称列表，如果为 None 则加载所有特征
            force: 是否强制重新计算，忽略缓存
            
        Returns:
            DataFrame，索引为日期，列为特征名称
            所有 null 值（包括节假日）都会被插值补全
        """
        # 先加载原始数据
        raw_loader = dataloader_raw(self.symbol)
        result = raw_loader.load(start_date, end_date, feature_names, force=force)
        
        # 对每个特征列进行插值补全（包括节假日）
        for col in result.columns:
            # 使用前向填充和后向填充进行插值
            # 先前向填充
            result[col] = result[col].ffill()
            # 再后向填充
            result[col] = result[col].bfill()
            
            # 如果还有缺失值（例如开始或结束日期是节假日），使用线性插值
            if result[col].isna().any():
                result[col] = result[col].interpolate(method='linear', limit_direction='both')
        
        logger.info(f"dataloader_autocomplete: 加载了 {len(result)} 个日期的数据，"
                   f"所有 null 值（包括节假日）都已补全")
        
        return result

