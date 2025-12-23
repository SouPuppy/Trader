"""
dataloader_raw: 原始数据加载器
不进行任何补全，节假日返回 None
"""
from typing import Optional, List
import pandas as pd
from trader.dataloader.dataloader import Dataloader
from trader.logger import get_logger

logger = get_logger(__name__)


class dataloader_raw(Dataloader):
    """
    原始数据加载器
    不进行任何补全，节假日返回 None
    """
    
    def load(self, start_date: str, end_date: str, feature_names: Optional[List[str]] = None, 
             force: bool = False) -> pd.DataFrame:
        """
        加载从开始日期到结束日期的所有 features（原始数据，不补全）
        
        Args:
            start_date: 开始日期 (YYYY-MM-DD)
            end_date: 结束日期 (YYYY-MM-DD)
            feature_names: 要加载的特征名称列表，如果为 None 则加载所有特征
            force: 是否强制重新计算，忽略缓存
            
        Returns:
            DataFrame，索引为日期，列为特征名称
            节假日返回 None（在对应日期行中）
        """
        # 获取所有日期范围
        date_range = self._get_date_range(start_date, end_date)
        
        # 获取所有特征名称
        if feature_names is None:
            from trader.features.registry import get_feature_names
            feature_names = get_feature_names()
        
        # 初始化结果 DataFrame（确保包含所有日期）
        result = pd.DataFrame(index=date_range, columns=feature_names, dtype=float)
        result.index.name = 'date'
        
        # 为每个日期加载特征（确保遍历所有日期）
        for date in date_range:
            date_str = date.strftime('%Y-%m-%d')
            
            # 加载该日期的所有特征
            features = self._load_features_for_date(date_str, force=force)
            
            if features is None:
                # 节假日，所有特征设为 None（使用 pd.NA 而不是 None）
                result.loc[date] = pd.NA
            else:
                # 交易日，填充特征值（包括 None 值）
                for feature_name in feature_names:
                    if feature_name in features:
                        value = features[feature_name]
                        result.loc[date, feature_name] = value if value is not None else pd.NA
                    else:
                        result.loc[date, feature_name] = pd.NA
        
        logger.info(f"dataloader_raw: 加载了 {len(result)} 个日期的数据，"
                   f"其中 {result.isna().all(axis=1).sum()} 个节假日")
        
        return result

