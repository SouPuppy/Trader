"""
特征定义
定义所有可用的特征计算函数
"""
import pandas as pd
from typing import Optional
from pathlib import Path
import sys

# 添加项目根目录到 Python 路径
project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from trader.features.registry import FeatureSpec, register
from trader.config import DB_PATH
from trader.logger import get_logger

logger = get_logger(__name__)


def load_stock_data(symbol: str, lookback: int = 0) -> pd.DataFrame:
    """
    从数据库加载股票数据
    
    Args:
        symbol: 股票代码
        lookback: 回看窗口大小（需要多少天的历史数据）
        
    Returns:
        DataFrame，包含股票数据，按日期排序
    """
    import sqlite3
    
    if not DB_PATH.exists():
        logger.error(f"数据库文件不存在: {DB_PATH}")
        return pd.DataFrame()
    
    try:
        conn = sqlite3.connect(DB_PATH)
        
        # 如果 lookback > 0，需要加载历史数据
        if lookback > 0:
            query = """
                SELECT 
                    datetime,
                    stock_code,
                    prev_close,
                    open_price,
                    high_price,
                    low_price,
                    close_price,
                    volume,
                    pe_ratio,
                    pe_ratio_ttm,
                    pcf_ratio_ttm,
                    pb_ratio,
                    ps_ratio,
                    ps_ratio_ttm
                FROM raw_data 
                WHERE stock_code = ?
                ORDER BY datetime DESC
                LIMIT ?
            """
            df = pd.read_sql_query(query, conn, params=(symbol, lookback + 1))
        else:
            # 只加载最新的一条记录
            query = """
                SELECT 
                    datetime,
                    stock_code,
                    prev_close,
                    open_price,
                    high_price,
                    low_price,
                    close_price,
                    volume,
                    pe_ratio,
                    pe_ratio_ttm,
                    pcf_ratio_ttm,
                    pb_ratio,
                    ps_ratio,
                    ps_ratio_ttm
                FROM raw_data 
                WHERE stock_code = ?
                ORDER BY datetime DESC
                LIMIT 1
            """
            df = pd.read_sql_query(query, conn, params=(symbol,))
        
        conn.close()
        
        if df.empty:
            logger.warning(f"未找到股票数据: symbol={symbol}")
            return pd.DataFrame()
        
        # 确保 datetime 是 datetime 类型
        if 'datetime' in df.columns:
            df['datetime'] = pd.to_datetime(df['datetime'])
            df = df.sort_values('datetime')
        
        return df
        
    except Exception as e:
        logger.error(f"加载股票数据时出错: {e}", exc_info=True)
        return pd.DataFrame()


def create_simple_feature_compute(column_name: str):
    """
    创建一个简单的特征计算函数
    直接从 DataFrame 的指定列读取值
    
    Args:
        column_name: 数据库列名
        
    Returns:
        计算函数，接受 DataFrame 返回 Series
    """
    def compute(df: pd.DataFrame) -> pd.Series:
        """
        从 DataFrame 读取指定列的值
        
        Args:
            df: 包含股票数据的 DataFrame
            
        Returns:
            Series，包含特征值，索引与 DataFrame 相同
        """
        if df.empty or column_name not in df.columns:
            return pd.Series(dtype='float64', index=df.index)
        
        # 返回该列的所有值（保持与 DataFrame 相同的索引）
        return df[column_name]
    
    return compute


# ============================================================================
# 特征定义
# ============================================================================

# 财务比率特征
register(FeatureSpec(
    name='pe_ratio',
    dtype='float64',
    lookback=0,
    compute=create_simple_feature_compute('pe_ratio'),
    desc='市盈率（PE Ratio）'
))

register(FeatureSpec(
    name='pe_ratio_ttm',
    dtype='float64',
    lookback=0,
    compute=create_simple_feature_compute('pe_ratio_ttm'),
    desc='滚动市盈率（PE Ratio TTM）'
))

register(FeatureSpec(
    name='pcf_ratio_ttm',
    dtype='float64',
    lookback=0,
    compute=create_simple_feature_compute('pcf_ratio_ttm'),
    desc='市现率TTM（PCF Ratio TTM）'
))

register(FeatureSpec(
    name='pb_ratio',
    dtype='float64',
    lookback=0,
    compute=create_simple_feature_compute('pb_ratio'),
    desc='市净率（PB Ratio）'
))

register(FeatureSpec(
    name='ps_ratio',
    dtype='float64',
    lookback=0,
    compute=create_simple_feature_compute('ps_ratio'),
    desc='市销率（PS Ratio）'
))

register(FeatureSpec(
    name='ps_ratio_ttm',
    dtype='float64',
    lookback=0,
    compute=create_simple_feature_compute('ps_ratio_ttm'),
    desc='滚动市销率（PS Ratio TTM）'
))

# 价格特征
register(FeatureSpec(
    name='prev_close',
    dtype='float64',
    lookback=0,
    compute=create_simple_feature_compute('prev_close'),
    desc='昨日收盘价（Previous Close）'
))

register(FeatureSpec(
    name='open_price',
    dtype='float64',
    lookback=0,
    compute=create_simple_feature_compute('open_price'),
    desc='开盘价（Open Price）'
))

register(FeatureSpec(
    name='high_price',
    dtype='float64',
    lookback=0,
    compute=create_simple_feature_compute('high_price'),
    desc='最高价（High Price）'
))

register(FeatureSpec(
    name='low_price',
    dtype='float64',
    lookback=0,
    compute=create_simple_feature_compute('low_price'),
    desc='最低价（Low Price）'
))

register(FeatureSpec(
    name='close_price',
    dtype='float64',
    lookback=0,
    compute=create_simple_feature_compute('close_price'),
    desc='收盘价（Close Price）'
))

# 交易量特征
register(FeatureSpec(
    name='volume',
    dtype='float64',
    lookback=0,
    compute=create_simple_feature_compute('volume'),
    desc='成交量（Volume）'
))

