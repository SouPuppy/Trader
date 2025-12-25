"""
Query 格式定义
TradingAgent 的查询格式：向量检索 + 硬过滤
"""
from dataclasses import dataclass
from typing import Optional, List, Dict, Any
from datetime import datetime, timedelta


@dataclass
class QueryFilter:
    """
    硬过滤条件（必须有）
    用于确保回测正确性（防止未来信息泄漏）
    """
    symbol: Optional[str] = None  # symbol == X（如果是单标的策略）
    time_max: Optional[str] = None  # time <= now（回测必备）
    time_min: Optional[str] = None  # time >= now-7d（新闻常用）
    source: Optional[str] = None  # source == X
    metadata_filters: Dict[str, Any] = None  # 额外的metadata过滤条件
    
    def __post_init__(self):
        if self.metadata_filters is None:
            self.metadata_filters = {}
    
    def to_sql_where(self) -> tuple[str, List[Any]]:
        """
        转换为SQL WHERE子句
        返回: (where_clause, params)
        """
        conditions = []
        params = []
        
        if self.symbol is not None:
            conditions.append("symbol = ?")
            params.append(self.symbol)
        
        if self.time_max is not None:
            conditions.append("time <= ?")
            params.append(self.time_max)
        
        if self.time_min is not None:
            conditions.append("time >= ?")
            params.append(self.time_min)
        
        if self.source is not None:
            conditions.append("source = ?")
            params.append(self.source)
        
        # metadata过滤（简单实现：只支持相等匹配）
        for key, value in self.metadata_filters.items():
            # 使用JSON_EXTRACT进行metadata字段过滤
            conditions.append(f"JSON_EXTRACT(metadata, '$.{key}') = ?")
            params.append(value)
        
        if conditions:
            where_clause = " AND ".join(conditions)
            return f"WHERE {where_clause}", params
        else:
            return "", []


@dataclass
class Query:
    """
    查询格式
    
    核心原则：
    - query_text: 用于向量检索的文字
    - filters: 硬过滤条件（必须有）
    - top_k: 返回数量（比如5或10）
    - ranking_policy: 相似度 + 时间衰减（近的更重要）
    
    向量检索负责"相关性"，过滤负责"正确性"
    """
    query_text: str  # 用于向量检索的文字
    filters: QueryFilter  # 硬过滤条件（必须有）
    top_k: int = 5  # 返回数量
    ranking_policy: str = "similarity_time_decay"  # 排序策略：similarity_time_decay / similarity_only / time_only
    
    @classmethod
    def create_for_backtest(
        cls,
        query_text: str,
        current_time: str,
        symbol: Optional[str] = None,
        lookback_days: int = 7,
        top_k: int = 5,
    ) -> 'Query':
        """
        创建回测查询（自动设置时间过滤，防止未来信息泄漏）
        
        Args:
            query_text: 查询文本
            current_time: 当前时间（格式：YYYY-MM-DD HH:MM:SS）
            symbol: 标的（可选）
            lookback_days: 回溯天数（默认7天）
            top_k: 返回数量
        """
        # 计算时间范围
        time_max = current_time
        time_min = (datetime.strptime(current_time, "%Y-%m-%d %H:%M:%S") - timedelta(days=lookback_days)).strftime("%Y-%m-%d %H:%M:%S")
        
        filters = QueryFilter(
            symbol=symbol,
            time_max=time_max,
            time_min=time_min,
        )
        
        return cls(
            query_text=query_text,
            filters=filters,
            top_k=top_k,
            ranking_policy="similarity_time_decay",
        )
    
    @classmethod
    def create_for_news(
        cls,
        query_text: str,
        current_time: str,
        symbol: Optional[str] = None,
        lookback_days: int = 7,
        top_k: int = 10,
    ) -> 'Query':
        """
        创建新闻查询（默认7天回溯）
        """
        return cls.create_for_backtest(
            query_text=query_text,
            current_time=current_time,
            symbol=symbol,
            lookback_days=lookback_days,
            top_k=top_k,
        )


