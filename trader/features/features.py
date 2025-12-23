"""
特征定义
定义所有可用的特征计算函数
"""
import pandas as pd
from typing import Optional
from pathlib import Path
import sys

try:
    from tqdm import tqdm
except ImportError:
    # 如果 tqdm 未安装，使用一个简单的替代实现
    def tqdm(iterable, desc=None, total=None, **kwargs):
        if desc:
            print(f"{desc}...")
        return iterable

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

# ============================================================================
# 收益率特征（Returns）
# ============================================================================

def compute_ret_1d(df: pd.DataFrame) -> pd.Series:
    """1日收益率: close_price / prev_close - 1"""
    if df.empty or 'close_price' not in df.columns or 'prev_close' not in df.columns:
        return pd.Series(dtype='float64', index=df.index)
    
    # 处理缺失值：forward fill
    close_price = df['close_price'].ffill()
    prev_close = df['prev_close'].ffill()
    
    result = close_price / prev_close - 1
    return result.replace([float('inf'), float('-inf')], pd.NA)


register(FeatureSpec(
    name='ret_1d',
    dtype='float64',
    lookback=1,  # 需要前一日数据
    compute=compute_ret_1d,
    deps=['close_price', 'prev_close'],
    desc='1日收益率（1-day Return）'
))


def compute_ret_5d(df: pd.DataFrame) -> pd.Series:
    """5日收益率: close_price / close_price.shift(5) - 1"""
    if df.empty or 'close_price' not in df.columns:
        return pd.Series(dtype='float64', index=df.index)
    
    close_price = df['close_price'].ffill()
    close_5d_ago = close_price.shift(5)
    
    result = close_price / close_5d_ago - 1
    return result.replace([float('inf'), float('-inf')], pd.NA)


register(FeatureSpec(
    name='ret_5d',
    dtype='float64',
    lookback=5,  # 需要5天前数据
    compute=compute_ret_5d,
    deps=['close_price'],
    desc='5日收益率（5-day Return）'
))


def compute_ret_20d(df: pd.DataFrame) -> pd.Series:
    """20日收益率: close_price / close_price.shift(20) - 1"""
    if df.empty or 'close_price' not in df.columns:
        return pd.Series(dtype='float64', index=df.index)
    
    close_price = df['close_price'].ffill()
    close_20d_ago = close_price.shift(20)
    
    result = close_price / close_20d_ago - 1
    return result.replace([float('inf'), float('-inf')], pd.NA)


register(FeatureSpec(
    name='ret_20d',
    dtype='float64',
    lookback=20,  # 需要20天前数据
    compute=compute_ret_20d,
    deps=['close_price'],
    desc='20日收益率（20-day Return）'
))

# ============================================================================
# 日内波动/跳空特征（Intraday）
# ============================================================================

def compute_range_pct(df: pd.DataFrame) -> pd.Series:
    """日内波动率: (high_price - low_price) / prev_close"""
    if df.empty or 'high_price' not in df.columns or 'low_price' not in df.columns or 'prev_close' not in df.columns:
        return pd.Series(dtype='float64', index=df.index)
    
    high_price = df['high_price'].ffill()
    low_price = df['low_price'].ffill()
    prev_close = df['prev_close'].ffill()
    
    result = (high_price - low_price) / prev_close
    return result.replace([float('inf'), float('-inf')], pd.NA)


register(FeatureSpec(
    name='range_pct',
    dtype='float64',
    lookback=1,  # 需要前一日收盘价
    compute=compute_range_pct,
    deps=['high_price', 'low_price', 'prev_close'],
    desc='日内波动率（Intraday Range Percentage）'
))


def compute_gap_pct(df: pd.DataFrame) -> pd.Series:
    """跳空幅度: (open_price - prev_close) / prev_close"""
    if df.empty or 'open_price' not in df.columns or 'prev_close' not in df.columns:
        return pd.Series(dtype='float64', index=df.index)
    
    open_price = df['open_price'].ffill()
    prev_close = df['prev_close'].ffill()
    
    result = (open_price - prev_close) / prev_close
    return result.replace([float('inf'), float('-inf')], pd.NA)


register(FeatureSpec(
    name='gap_pct',
    dtype='float64',
    lookback=1,  # 需要前一日收盘价
    compute=compute_gap_pct,
    deps=['open_price', 'prev_close'],
    desc='跳空幅度（Gap Percentage）'
))


def compute_close_to_open(df: pd.DataFrame) -> pd.Series:
    """收盘相对开盘: close_price / open_price - 1"""
    if df.empty or 'close_price' not in df.columns or 'open_price' not in df.columns:
        return pd.Series(dtype='float64', index=df.index)
    
    close_price = df['close_price'].ffill()
    open_price = df['open_price'].ffill()
    
    result = close_price / open_price - 1
    return result.replace([float('inf'), float('-inf')], pd.NA)


register(FeatureSpec(
    name='close_to_open',
    dtype='float64',
    lookback=0,
    compute=compute_close_to_open,
    deps=['close_price', 'open_price'],
    desc='收盘相对开盘（Close to Open）'
))

# ============================================================================
# 波动率与成交量特征（Volatility & Liquidity）
# ============================================================================

def compute_vol_20d(df: pd.DataFrame) -> pd.Series:
    """20日波动率: rolling 20 日 ret_1d 标准差"""
    if df.empty or 'close_price' not in df.columns or 'prev_close' not in df.columns:
        return pd.Series(dtype='float64', index=df.index)
    
    # 先计算 ret_1d
    close_price = df['close_price'].ffill()
    prev_close = df['prev_close'].ffill()
    ret_1d = close_price / prev_close - 1
    
    # 计算滚动标准差
    vol_20d = ret_1d.rolling(window=20, min_periods=1).std()
    return vol_20d.replace([float('inf'), float('-inf')], pd.NA)


register(FeatureSpec(
    name='vol_20d',
    dtype='float64',
    lookback=20,  # 需要20天历史数据
    compute=compute_vol_20d,
    deps=['close_price', 'prev_close'],
    desc='20日波动率（20-day Volatility）'
))


def compute_vol_60d(df: pd.DataFrame) -> pd.Series:
    """60日波动率: rolling 60 日 ret_1d 标准差"""
    if df.empty or 'close_price' not in df.columns or 'prev_close' not in df.columns:
        return pd.Series(dtype='float64', index=df.index)
    
    # 先计算 ret_1d
    close_price = df['close_price'].ffill()
    prev_close = df['prev_close'].ffill()
    ret_1d = close_price / prev_close - 1
    
    # 计算滚动标准差
    vol_60d = ret_1d.rolling(window=60, min_periods=1).std()
    return vol_60d.replace([float('inf'), float('-inf')], pd.NA)


register(FeatureSpec(
    name='vol_60d',
    dtype='float64',
    lookback=60,  # 需要60天历史数据
    compute=compute_vol_60d,
    deps=['close_price', 'prev_close'],
    desc='60日波动率（60-day Volatility）'
))


def compute_vol_z_20d(df: pd.DataFrame) -> pd.Series:
    """20日成交量 z-score: (volume - volume_20d_mean) / volume_20d_std"""
    if df.empty or 'volume' not in df.columns:
        return pd.Series(dtype='float64', index=df.index)
    
    volume = df['volume'].ffill()
    
    # 计算滚动均值和标准差
    volume_mean = volume.rolling(window=20, min_periods=1).mean()
    volume_std = volume.rolling(window=20, min_periods=1).std()
    
    # 计算 z-score，避免除零
    volume_std_safe = volume_std.replace(0, pd.NA)
    vol_z_20d = (volume - volume_mean) / volume_std_safe
    return vol_z_20d.replace([float('inf'), float('-inf')], pd.NA)


register(FeatureSpec(
    name='vol_z_20d',
    dtype='float64',
    lookback=20,  # 需要20天历史数据
    compute=compute_vol_z_20d,
    deps=['volume'],
    desc='20日成交量 z-score（20-day Volume Z-score）'
))

# ============================================================================
# 新闻特征（News Features）
# ============================================================================

def load_news_data_for_dates(symbol: str, dates: list) -> pd.DataFrame:
    """
    从 raw_data 表的 news 字段加载指定日期的新闻数据（已废弃）
    
    注意：此函数使用现场分析，已废弃。请使用 compute_news_features_by_sql 函数，
    它直接从 news_data 表查询 sentiment 和 impact。
    
    Args:
        symbol: 股票代码
        dates: 日期列表（字符串格式：'YYYY-MM-DD'）
        
    Returns:
        DataFrame，包含新闻数据，按日期分组
        列：datetime, sentiment, impact
    """
    import sqlite3
    
    if not DB_PATH.exists() or not dates:
        return pd.DataFrame(columns=['datetime', 'sentiment', 'impact'])
    
    try:
        from trader.news.prepare import parse_news_json
        from trader.news.analyze import analyze
        
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        # 查询 raw_data 表中相关日期的新闻数据
        dates_placeholder = ','.join(['?'] * len(dates))
        query = f"""
            SELECT datetime, news
            FROM raw_data
            WHERE stock_code = ?
              AND datetime IN ({dates_placeholder})
              AND news IS NOT NULL 
              AND news != ''
        """
        cursor.execute(query, (symbol, *dates))
        rows = cursor.fetchall()
        
        result_rows = []
        for row in rows:
            date_str = row['datetime']
            news_json = row['news']
            
            if not news_json:
                continue
            
            # 解析新闻 JSON
            news_obj = parse_news_json(news_json)
            if not news_obj:
                continue
            
            # 如果是列表，处理每个新闻项
            if isinstance(news_obj, list):
                news_list = news_obj
            else:
                news_list = [news_obj]
            
            # 分析每个新闻
            for news_item in news_list:
                if not isinstance(news_item, dict):
                    continue
                
                # 构建新闻字典用于分析
                news_dict = {
                    'publish_time': news_item.get('publish_time', ''),
                    'title': news_item.get('title', ''),
                    'content': news_item.get('content', '')
                }
                
                if not news_dict['title'] and not news_dict['content']:
                    continue
                
                # 分析新闻（获取 sentiment 和 impact）
                try:
                    analysis_result = analyze(news_dict)
                    if not analysis_result:
                        # 如果分析失败，使用默认值
                        sentiment = 0
                        impact = 0
                    else:
                        sentiment = analysis_result.get('sentiment', 0)
                        impact = analysis_result.get('impact', 0)
                    
                    result_rows.append({
                        'datetime': date_str,
                        'sentiment': sentiment,
                        'impact': impact
                    })
                except Exception as e:
                    # 如果分析失败（例如未安装 openai），使用默认值
                    logger.debug(f"分析新闻失败 (date={date_str}): {e}，使用默认值")
                    result_rows.append({
                        'datetime': date_str,
                        'sentiment': 0,
                        'impact': 0
                    })
                    continue
        
        conn.close()
        
        if not result_rows:
            return pd.DataFrame(columns=['datetime', 'sentiment', 'impact'])
        
        result_df = pd.DataFrame(result_rows)
        result_df['datetime'] = pd.to_datetime(result_df['datetime'])
        result_df = result_df.sort_values('datetime')
        
        return result_df
        
    except Exception as e:
        logger.error(f"加载新闻数据时出错: {e}", exc_info=True)
        return pd.DataFrame(columns=['datetime', 'sentiment', 'impact'])


def compute_news_features_by_sql(symbol: str, dates: list, feature_type: str, window: int = 0) -> pd.DataFrame:
    """
    从 news_data 表聚合查询新闻特征（不再现场分析）
    
    首先从 raw_data 获取 stock_code 和 datetime，然后根据日期关联 news_data 表，
    聚合 news_data 的 sentiment 和 impact 进行统计。
    
    Args:
        symbol: 股票代码
        dates: 日期列表（字符串格式：'YYYY-MM-DD'）
        feature_type: 特征类型 ('count', 'sentiment_mean', 'impact_mean', 'impact_sum', 'weighted_sentiment')
        window: 时间窗口（0 表示当日，>0 表示近期N天）
        
    Returns:
        DataFrame，包含日期和特征值
        列：datetime, feature_value
    """
    import sqlite3
    
    if not DB_PATH.exists() or not dates:
        logger.debug(f"数据库不存在或日期列表为空: symbol={symbol}, feature_type={feature_type}")
        return pd.DataFrame(columns=['datetime', 'feature_value'])
    
    logger.debug(f"开始计算新闻特征: symbol={symbol}, feature_type={feature_type}, window={window}, dates_count={len(dates)}")
    
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        # 验证 feature_type
        valid_types = ['count', 'sentiment_mean', 'impact_mean', 'impact_sum', 'weighted_sentiment']
        if feature_type not in valid_types:
            logger.warning(f"未知的特征类型: {feature_type}")
            conn.close()
            return pd.DataFrame(columns=['datetime', 'feature_value'])
        
        # 计算需要查询的日期范围（包括窗口）
        if window > 0:
            all_dates_set = set()
            for date_str in dates:
                date_obj = pd.to_datetime(date_str)
                for i in range(window + 1):
                    window_date = (date_obj - pd.Timedelta(days=i)).strftime('%Y-%m-%d')
                    all_dates_set.add(window_date)
            query_dates = sorted(list(all_dates_set))
        else:
            query_dates = dates
        
        # 首先从 raw_data 获取该股票的所有相关日期记录（包括没有新闻的日期）
        # 然后通过这些 publish_time 精确匹配 news_data 表（避免所有股票得到相同结果）
        dates_placeholder = ','.join(['?'] * len(query_dates))
        # 先查询所有日期的记录（不管有没有新闻）
        raw_data_query_all = f"""
            SELECT datetime, news
            FROM raw_data
            WHERE stock_code = ?
              AND datetime IN ({dates_placeholder})
        """
        cursor.execute(raw_data_query_all, (symbol, *query_dates))
        raw_data_rows_all = cursor.fetchall()
        
        # 建立日期到记录的映射，确保所有日期都有记录（即使没有新闻）
        date_to_raw_data = {}
        for row in raw_data_rows_all:
            date_str = row[0]
            news_json = row[1]
            if date_str not in date_to_raw_data:
                date_to_raw_data[date_str] = []
            date_to_raw_data[date_str].append(news_json)
        
        # 解析 raw_data 中的 news，提取 publish_time（只解析，不分析）
        from trader.news.prepare import parse_news_json
        publish_times = set()  # 使用 set 去重
        date_to_publish_times = {}  # 日期 -> publish_time 列表
        
        # 遍历所有日期，处理有新闻的日期
        for date_str, news_json_list in date_to_raw_data.items():
            for news_json in news_json_list:
                if not news_json:
                    continue
                
                # 解析新闻 JSON（只解析，不分析）
                # 临时禁用 trader.news.prepare 模块的日志，避免大量警告
                import logging
                news_prepare_logger = logging.getLogger('trader.news.prepare')
                old_level = news_prepare_logger.level
                news_prepare_logger.setLevel(logging.ERROR)  # 只显示 ERROR 级别以上的日志
                
                try:
                    news_obj = parse_news_json(news_json)
                except Exception:
                    # 解析失败，跳过这条新闻
                    news_obj = None
                finally:
                    news_prepare_logger.setLevel(old_level)  # 恢复原来的日志级别
                
                if not news_obj:
                    continue
                
                # 如果是列表，处理每个新闻项
                if isinstance(news_obj, list):
                    news_list = news_obj
                else:
                    news_list = [news_obj]
                
                # 提取 publish_time
                for news_item in news_list:
                    if not isinstance(news_item, dict):
                        continue
                    
                    publish_time = news_item.get('publish_time', '')
                    if publish_time:
                        publish_times.add(publish_time)
                        if date_str not in date_to_publish_times:
                            date_to_publish_times[date_str] = []
                        date_to_publish_times[date_str].append(publish_time)
        
        # 注意：即使没有 publish_times，我们仍然需要为所有日期计算特征
        # 因为有些日期可能没有新闻，应该返回 0
        
        # 查询 news_data 表中匹配这些 publish_time 的新闻数据
        # 注意：news_data 的 publish_date 格式为 'YYYY-MM-DD HH:MM:SS UTC'
        # 需要匹配 publish_time（格式可能为 'YYYY-MM-DD HH:MM:SS'）
        publish_time_to_data_list = {}
        
        if publish_times:
            publish_time_conditions = []
            publish_time_params = []
            
            for pt in publish_times:
                # 将 publish_time 转换为 UTC 格式进行匹配
                # 简化处理：直接匹配日期和时间部分
                publish_time_conditions.append("publish_date LIKE ?")
                # publish_time 可能是 'YYYY-MM-DD HH:MM:SS' 格式，需要匹配 'YYYY-MM-DD HH:MM:SS%'
                publish_time_params.append(f"{pt}%")
            
            news_data_query = f"""
                SELECT publish_date, sentiment, impact
                FROM news_data
                WHERE ({' OR '.join(publish_time_conditions)})
                  AND sentiment IS NOT NULL
                  AND impact IS NOT NULL
            """
            cursor.execute(news_data_query, publish_time_params)
            news_data_rows = cursor.fetchall()
            
            # 建立 publish_time -> [(sentiment, impact), ...] 的映射
            # 注意：只匹配该股票在 raw_data 中出现的 publish_time，确保每个股票只统计自己的新闻
            for row in news_data_rows:
                publish_date = row[0]
                sentiment = row[1]
                impact = row[2]
                # publish_date 格式：'YYYY-MM-DD HH:MM:SS UTC'
                # 提取前19个字符（去掉 ' UTC' 后缀）
                publish_date_key = publish_date[:19] if len(publish_date) >= 19 else publish_date  # 'YYYY-MM-DD HH:MM:SS'
                
                # 只匹配该股票在 raw_data 中出现的 publish_time
                matched_pt = None
                for pt in publish_times:
                    # 精确匹配：publish_date_key 应该以 publish_time 开头
                    # 或者匹配到分钟级别（处理秒数缺失的情况）
                    if publish_date_key.startswith(pt):
                        matched_pt = pt
                        break
                    elif len(pt) >= 16 and len(publish_date_key) >= 16 and publish_date_key[:16] == pt[:16]:
                        # 匹配到分钟级别
                        matched_pt = pt
                        break
                
                if matched_pt:
                    if matched_pt not in publish_time_to_data_list:
                        publish_time_to_data_list[matched_pt] = []
                    publish_time_to_data_list[matched_pt].append((sentiment, impact))
        
        result_rows = []
        
        # 为每个目标日期计算特征
        for date_str in dates:
            # 确定查询的日期范围
            if window > 0:
                date_obj = pd.to_datetime(date_str)
                window_dates = []
                for i in range(window + 1):
                    window_date = (date_obj - pd.Timedelta(days=i)).strftime('%Y-%m-%d')
                    window_dates.append(window_date)
            else:
                window_dates = [date_str]
            
            # 收集该日期范围内的所有新闻数据
            sentiments = []
            impacts = []
            
            for wd in window_dates:
                # 从 date_to_publish_times 获取该日期的 publish_time 列表
                # 注意：同一个 publish_time 可能在列表中多次出现（如果 raw_data 中该日期有多个记录包含相同的 publish_time）
                # 这种情况下，我们应该统计多次，因为同一个新闻可能在多个记录中出现
                if wd in date_to_publish_times:
                    for publish_time in date_to_publish_times[wd]:
                        # 直接从映射中获取匹配的新闻数据
                        if publish_time in publish_time_to_data_list:
                            # 遍历该 publish_time 对应的所有新闻数据
                            # 注意：同一个 publish_time 可能对应多条新闻（如果 news_data 表中有多条记录）
                            for sentiment, impact in publish_time_to_data_list[publish_time]:
                                sentiments.append(sentiment)
                                impacts.append(impact)
            
            if not sentiments:
                # 没有新闻数据，设置默认值（使用 0.0 而不是 pd.NA，确保有数据点）
                default_value = 0.0
                result_rows.append({'datetime': date_str, 'feature_value': default_value})
                continue
            
            # 聚合计算
            news_count = len(sentiments)
            
            if feature_type == 'count':
                feature_value = news_count
            elif feature_type == 'sentiment_mean':
                feature_value = sum(sentiments) / len(sentiments) if sentiments else 0.0
            elif feature_type == 'impact_mean':
                feature_value = sum(impacts) / len(impacts) if impacts else 0.0
            elif feature_type == 'impact_sum':
                feature_value = sum(impacts) if impacts else 0.0
            elif feature_type == 'weighted_sentiment':
                # 加权情绪：sum(sentiment * impact) / sum(impact)
                weighted_sum = sum(s * i for s, i in zip(sentiments, impacts))
                impact_sum = sum(impacts)
                feature_value = weighted_sum / impact_sum if impact_sum > 0 else 0.0
            else:
                default_value = 0.0
                feature_value = default_value
            
            result_rows.append({
                'datetime': date_str,
                'feature_value': feature_value
            })
        
        conn.close()
        
        if not result_rows:
            return pd.DataFrame(columns=['datetime', 'feature_value'])
        
        result_df = pd.DataFrame(result_rows)
        result_df['datetime'] = pd.to_datetime(result_df['datetime'])
        result_df = result_df.sort_values('datetime')
        
        # 统计结果
        non_null_count = result_df['feature_value'].notna().sum()
        logger.debug(f"计算完成: {symbol} {feature_type} (window={window}) - 共 {len(result_df)} 个日期，{non_null_count} 个有值")
        
        return result_df
        
    except Exception as e:
        logger.error(f"计算新闻特征时出错 (symbol={symbol}, feature_type={feature_type}, window={window}): {e}", exc_info=True)
        return pd.DataFrame(columns=['datetime', 'feature_value'])






def create_news_feature_compute(feature_type: str, window: int = 0):
    """
    创建新闻特征计算函数（从 raw_data 表的 news 字段解析 JSON）
    
    Args:
        feature_type: 特征类型 ('count', 'sentiment_mean', 'impact_mean', 'impact_sum', 'weighted_sentiment')
        window: 时间窗口（0 表示当日，>0 表示近期N天）
    """
    def compute(df: pd.DataFrame) -> pd.Series:
        """
        计算新闻特征（通过 SQL 直接计算）
        
        Args:
            df: 包含股票数据的 DataFrame
            
        Returns:
            Series，包含特征值
        """
        if df.empty:
            return pd.Series(dtype='float64', index=df.index)
        
        # 处理 datetime 可能是索引或列的情况
        if isinstance(df.index, pd.DatetimeIndex):
            # datetime 是索引
            has_datetime_col = False
        elif 'datetime' in df.columns:
            # datetime 是列
            has_datetime_col = True
        else:
            logger.warning(f"无法找到 datetime（既不是索引也不是列）")
            return pd.Series(dtype='float64', index=df.index)
        
        # 处理 stock_code
        if 'stock_code' not in df.columns:
            logger.warning(f"无法找到 stock_code 列")
            return pd.Series(dtype='float64', index=df.index)
        
        result = pd.Series(index=df.index, dtype='float64')
        
        # 获取股票代码（假设所有行的股票代码相同）
        stock_code = df['stock_code'].iloc[0] if not df.empty else None
        if not stock_code:
            return result
        
        # 收集所有需要查询的日期
        dates_to_query = []
        for idx in df.index:
            if has_datetime_col:
                date = df.loc[idx, 'datetime']
            else:
                date = idx  # datetime 是索引
            
            if pd.notna(date):
                date_str = pd.Timestamp(date).strftime('%Y-%m-%d')
                dates_to_query.append(date_str)
        
        if not dates_to_query:
            return result
        
        # 通过 SQL 直接计算特征
        try:
            dates_list = list(set(dates_to_query))
            logger.debug(f"计算新闻特征: symbol={stock_code}, feature_type={feature_type}, window={window}, dates={len(dates_list)}")
            feature_df = compute_news_features_by_sql(stock_code, dates_list, feature_type, window)
            
            # 设置默认值
            default_value = 0.0 if feature_type == 'count' or feature_type == 'impact_sum' else pd.NA
            
            if feature_df.empty:
                # 如果没有新闻数据，为所有日期设置默认值
                logger.debug(f"没有新闻数据: {stock_code}, 所有日期设置为默认值 {default_value}")
                for idx in df.index:
                    result[idx] = default_value
                return result
            
            # 将 SQL 查询结果映射回 DataFrame 的索引
            feature_dict = dict(zip(
                feature_df['datetime'].dt.strftime('%Y-%m-%d'),
                feature_df['feature_value']
            ))
            
            # 为每个索引位置设置特征值
            for idx in df.index:
                if has_datetime_col:
                    date = df.loc[idx, 'datetime']
                else:
                    date = idx  # datetime 是索引
                
                if pd.isna(date):
                    result[idx] = pd.NA
                    continue
                
                date_str = pd.Timestamp(date).strftime('%Y-%m-%d')
                # 如果该日期有数据，使用数据；否则使用默认值
                if date_str in feature_dict:
                    result[idx] = feature_dict[date_str]
                else:
                    result[idx] = default_value
                    
        except Exception as e:
            logger.error(f"通过 SQL 计算新闻特征时出错: {e}", exc_info=True)
            result[:] = pd.NA
        
        return result.replace([float('inf'), float('-inf')], pd.NA)
    
    return compute


# 注册新闻特征
register(FeatureSpec(
    name='news_count',
    dtype='float64',
    lookback=0,
    compute=create_news_feature_compute('count', window=0),
    desc='当日新闻数量（Daily News Count）'
))

# 不同时间窗口的新闻数量（窗口内新闻总数）
register(FeatureSpec(
    name='news_count_1d',
    dtype='float64',
    lookback=1,
    compute=create_news_feature_compute('count', window=1),
    desc='1日新闻数量（1-day News Count，包含当日和前1日）'
))

register(FeatureSpec(
    name='news_count_5d',
    dtype='float64',
    lookback=5,
    compute=create_news_feature_compute('count', window=5),
    desc='5日新闻数量（5-day News Count，包含当日和前4日）'
))

register(FeatureSpec(
    name='news_count_20d',
    dtype='float64',
    lookback=20,
    compute=create_news_feature_compute('count', window=20),
    desc='20日新闻数量（20-day News Count，包含当日和前19日）'
))

register(FeatureSpec(
    name='news_count_60d',
    dtype='float64',
    lookback=60,
    compute=create_news_feature_compute('count', window=60),
    desc='60日新闻数量（60-day News Count，包含当日和前59日）'
))

register(FeatureSpec(
    name='news_sentiment_mean',
    dtype='float64',
    lookback=0,
    compute=create_news_feature_compute('sentiment_mean', window=0),
    desc='当日新闻平均情绪（Daily News Sentiment Mean）'
))

register(FeatureSpec(
    name='news_impact_mean',
    dtype='float64',
    lookback=0,
    compute=create_news_feature_compute('impact_mean', window=0),
    desc='当日新闻平均影响强度（Daily News Impact Mean）'
))

register(FeatureSpec(
    name='news_impact_sum',
    dtype='float64',
    lookback=0,
    compute=create_news_feature_compute('impact_sum', window=0),
    desc='当日新闻影响强度总和（Daily News Impact Sum）'
))

register(FeatureSpec(
    name='news_weighted_sentiment',
    dtype='float64',
    lookback=0,
    compute=create_news_feature_compute('weighted_sentiment', window=0),
    desc='当日新闻加权情绪（Daily News Weighted Sentiment，按 impact 加权）'
))

# 近期新闻特征（7天窗口）
register(FeatureSpec(
    name='news_sentiment_mean_7d',
    dtype='float64',
    lookback=7,
    compute=create_news_feature_compute('sentiment_mean', window=7),
    desc='7日新闻平均情绪（7-day News Sentiment Mean）'
))

register(FeatureSpec(
    name='news_impact_mean_7d',
    dtype='float64',
    lookback=7,
    compute=create_news_feature_compute('impact_mean', window=7),
    desc='7日新闻平均影响强度（7-day News Impact Mean）'
))

register(FeatureSpec(
    name='news_weighted_sentiment_7d',
    dtype='float64',
    lookback=7,
    compute=create_news_feature_compute('weighted_sentiment', window=7),
    desc='7日新闻加权情绪（7-day News Weighted Sentiment）'
))

# 近期新闻特征（30天窗口）
register(FeatureSpec(
    name='news_sentiment_mean_30d',
    dtype='float64',
    lookback=30,
    compute=create_news_feature_compute('sentiment_mean', window=30),
    desc='30日新闻平均情绪（30-day News Sentiment Mean）'
))

register(FeatureSpec(
    name='news_impact_mean_30d',
    dtype='float64',
    lookback=30,
    compute=create_news_feature_compute('impact_mean', window=30),
    desc='30日新闻平均影响强度（30-day News Impact Mean）'
))

register(FeatureSpec(
    name='news_weighted_sentiment_30d',
    dtype='float64',
    lookback=30,
    compute=create_news_feature_compute('weighted_sentiment', window=30),
    desc='30日新闻加权情绪（30-day News Weighted Sentiment）'
))

