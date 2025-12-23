"""
dataloader_nullcomplete: Null 补全数据加载器
自动插值补全 null 值，但节假日仍返回 None
"""
from typing import Optional, List
import pandas as pd
import numpy as np
from trader.dataloader.dataloader import Dataloader
from trader.dataloader.dataloader_raw import dataloader_raw
from trader.logger import get_logger

logger = get_logger(__name__)


class dataloader_nullcomplete(Dataloader):
    """
    Null 补全数据加载器
    自动插值补全 null 值，但节假日仍返回 None
    """
    
    def load(self, start_date: str, end_date: str, feature_names: Optional[List[str]] = None, 
             force: bool = False) -> pd.DataFrame:
        """
        加载从开始日期到结束日期的所有 features（null 补全）
        
        Args:
            start_date: 开始日期 (YYYY-MM-DD)
            end_date: 结束日期 (YYYY-MM-DD)
            feature_names: 要加载的特征名称列表，如果为 None 则加载所有特征
            force: 是否强制重新计算，忽略缓存
            
        Returns:
            DataFrame，索引为日期，列为特征名称
            节假日返回 None（在对应日期行中），但交易日中的 null 值会被插值补全
        """
        # 先加载原始数据
        raw_loader = dataloader_raw(self.symbol)
        result = raw_loader.load(start_date, end_date, feature_names, force=force)
        
        # 标记哪些是节假日（整行都是 None）
        is_holiday = result.isna().all(axis=1)
        
        # 对每个特征列进行插值补全（但保留节假日的 None）
        for col in result.columns:
            # 只对交易日的数据进行插值
            trading_day_mask = ~is_holiday
            
            if trading_day_mask.sum() > 0:
                # 获取交易日的数据
                trading_data = result.loc[trading_day_mask, col].copy()
                
                # 使用前向填充和后向填充进行插值
                # 先前向填充
                trading_data_filled = trading_data.ffill()
                # 再后向填充
                trading_data_filled = trading_data_filled.bfill()
                
                # 如果还有缺失值，使用线性插值
                if trading_data_filled.isna().any():
                    trading_data_filled = trading_data_filled.interpolate(method='linear', limit_direction='both')
                
                # 更新结果（只更新交易日的数据）
                result.loc[trading_day_mask, col] = trading_data_filled
        
        logger.info(f"dataloader_nullcomplete: 加载了 {len(result)} 个日期的数据，"
                   f"其中 {is_holiday.sum()} 个节假日，"
                   f"交易日中的 null 值已补全")
        
        return result

