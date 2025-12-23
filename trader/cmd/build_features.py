"""
特征构建模块
使用 FeatureSpec 注册系统管理特征，同时提供便捷的 date/symbol 接口
"""
import sqlite3
from typing import Optional, Callable, Dict, Any
from pathlib import Path
import sys
import pandas as pd

import sys

try:
    from tqdm import tqdm
except ImportError:
    # 如果 tqdm 未安装，使用一个简单的替代实现
    def tqdm(iterable, desc=None, total=None, **kwargs):
        if desc:
            print(f"{desc}...", file=sys.stderr)
        return iterable

# 添加项目根目录到 Python 路径
project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from trader.config import DB_PATH
from trader.logger import get_logger
from trader.features.registry import FEATURES as REGISTRY_FEATURES, get_feature as get_feature_spec, get_all_features, get_feature_names
from trader.features import features  # 导入特征定义以触发注册
from trader.features.cache import get_cached_feature, cache_feature, ensure_features_table

logger = get_logger(__name__)


def load_stock_data_for_date(symbol: str, date: str, lookback: int = 0) -> pd.DataFrame:
    """
    从数据库加载指定日期和股票的数据
    
    Args:
        symbol: 股票代码
        date: 日期字符串，格式如 "2023-01-03"
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
                WHERE stock_code = ? AND datetime <= ?
                ORDER BY datetime DESC
                LIMIT ?
            """
            df = pd.read_sql_query(query, conn, params=(symbol, date, lookback + 1))
        else:
            # 只加载指定日期的记录
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
                WHERE stock_code = ? AND datetime = ?
                LIMIT 1
            """
            df = pd.read_sql_query(query, conn, params=(symbol, date))
        
        conn.close()
        
        if df.empty:
            logger.debug(f"未找到股票数据: symbol={symbol}, date={date}")
            return pd.DataFrame()
        
        # 确保 datetime 是 datetime 类型并设置为索引
        if 'datetime' in df.columns:
            df['datetime'] = pd.to_datetime(df['datetime'])
            df = df.sort_values('datetime')
            df = df.set_index('datetime')
        elif df.index.name != 'datetime' and not df.empty:
            # 如果没有 datetime 列，尝试使用索引
            if df.index.dtype == 'object':
                try:
                    df.index = pd.to_datetime(df.index)
                    df.index.name = 'datetime'
                except:
                    pass
        
        return df
        
    except Exception as e:
        logger.error(f"加载股票数据时出错: {e}", exc_info=True)
        return pd.DataFrame()


def compute_feature(feature_name: str, date: str, symbol: str, force: bool = False) -> Optional[float]:
    """
    计算指定特征的值（便捷接口）
    
    Args:
        feature_name: 特征名称
        date: 日期字符串，格式如 "2023-01-03"
        symbol: 股票代码，如 "AAPL.O"
        force: 是否强制重新计算，忽略缓存
        
    Returns:
        特征值（float），如果不存在则返回 None
    """
    logger.debug(f"计算特征: {feature_name} for {symbol} on {date}, force={force}")
    
    # 检查缓存（如果 force=False）
    if not force:
        cached_value = get_cached_feature(symbol, date, feature_name)
        if cached_value is not None:
            logger.debug(f"从缓存获取特征: {feature_name} for {symbol} on {date} = {cached_value}")
            return cached_value
    
    feature_spec = get_feature_spec(feature_name)
    if feature_spec is None:
        logger.warning(f"未知的特征名称: {feature_name}")
        return None
    
    # 加载数据
    logger.debug(f"加载股票数据: {symbol}, date={date}, lookback={feature_spec.lookback}")
    df = load_stock_data_for_date(symbol, date, lookback=feature_spec.lookback)
    if df.empty:
        logger.debug(f"未找到股票数据: {symbol} on {date}")
        return None
    
    logger.debug(f"加载了 {len(df)} 行数据")
    
    # 计算特征
    try:
        logger.debug(f"执行特征计算函数: {feature_name}")
        result_series = feature_spec.compute(df)
        if result_series.empty:
            logger.debug(f"特征计算结果为空: {feature_name}")
            return None
        # 返回最新值（最后一行，对应指定日期）
        # 如果 DataFrame 有 datetime 索引，尝试找到对应日期的值
        if 'datetime' in str(df.index.dtype) or isinstance(df.index, pd.DatetimeIndex):
            date_dt = pd.to_datetime(date)
            if date_dt in result_series.index:
                value = result_series.loc[date_dt]
            else:
                # 如果没有精确匹配，返回最新值
                value = result_series.iloc[-1]
        else:
            value = result_series.iloc[-1]
        
        result = float(value) if pd.notna(value) else None
        logger.debug(f"特征 {feature_name} 计算结果: {result}")
        
        # 存储到缓存
        cache_feature(symbol, date, feature_name, result)
        
        return result
    except Exception as e:
        logger.error(f"计算特征 {feature_name} 时出错: {e}", exc_info=True)
        return None


def compute_all_features(date: str, symbol: str, force: bool = False) -> Dict[str, Optional[float]]:
    """
    计算所有特征的值
    
    Args:
        date: 日期字符串，格式如 "2023-01-03"
        symbol: 股票代码，如 "AAPL.O"
        force: 是否强制重新计算，忽略缓存
        
    Returns:
        包含所有特征名称和值的字典
    """
    # 确保 features 表存在
    ensure_features_table()
    
    feature_names = get_feature_names()
    logger.info(f"开始计算所有特征: {symbol} on {date}, 共 {len(feature_names)} 个特征, force={force}")
    result = {}
    for feature_name in tqdm(feature_names, desc=f"计算特征 ({symbol})", unit="特征", file=sys.stderr, position=0, leave=True):
        result[feature_name] = compute_feature(feature_name, date, symbol, force=force)
    logger.info(f"特征计算完成: {symbol} on {date}, 共 {len(result)} 个特征")
    return result


def list_features() -> list:
    """
    列出所有可用的特征名称
    
    Returns:
        特征名称列表
    """
    return get_feature_names()


def get_feature_info(feature_name: str) -> Optional[Dict[str, Any]]:
    """
    获取特征的详细信息
    
    Args:
        feature_name: 特征名称
        
    Returns:
        特征信息字典，如果不存在则返回 None
    """
    feature_spec = get_feature_spec(feature_name)
    if feature_spec is None:
        return None
    
    return {
        'name': feature_spec.name,
        'dtype': feature_spec.dtype,
        'lookback': feature_spec.lookback,
        'deps': feature_spec.deps,
        'desc': feature_spec.desc,
    }


# ============================================================================
# 测试和示例
# ============================================================================

def main():
    """主函数，用于测试特征函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='特征构建模块测试')
    parser.add_argument('--date', type=str, help='日期，格式如 2023-01-03')
    parser.add_argument('--symbol', type=str, help='股票代码，如 AAPL.O')
    parser.add_argument('--feature', type=str, help='特征名称（可选，不指定则计算所有特征）')
    parser.add_argument('--list', action='store_true', help='列出所有可用的特征')
    parser.add_argument('--info', type=str, help='显示特征的详细信息')
    parser.add_argument('--force', action='store_true', help='强制重新计算，忽略缓存')
    
    args = parser.parse_args()
    
    if args.list:
        print("可用的特征列表:")
        print("-" * 60)
        for i, feature_name in enumerate(list_features(), 1):
            feature_spec = get_feature_spec(feature_name)
            desc = feature_spec.desc if feature_spec and feature_spec.desc else ""
            print(f"  {i:2d}. {feature_name:20s} - {desc}")
        return
    
    if args.info:
        info = get_feature_info(args.info)
        if info:
            print(f"特征信息: {args.info}")
            print("-" * 60)
            print(f"  名称:     {info['name']}")
            print(f"  类型:     {info['dtype']}")
            print(f"  回看窗口: {info['lookback']}")
            print(f"  依赖:     {info['deps'] or '无'}")
            print(f"  描述:     {info['desc']}")
        else:
            print(f"未找到特征: {args.info}")
        return
    
    if not args.date or not args.symbol:
        parser.print_help()
        print("\n示例:")
        print("  python -m trader.cmd.build_features --date 2023-01-03 --symbol AAPL.O")
        print("  python -m trader.cmd.build_features --date 2023-01-03 --symbol AAPL.O --feature pe_ratio")
        print("  python -m trader.cmd.build_features --list")
        print("  python -m trader.cmd.build_features --info pe_ratio")
        return
    
    if args.feature:
        # 计算单个特征
        value = compute_feature(args.feature, args.date, args.symbol, force=args.force)
        print(f"特征 {args.feature} (date={args.date}, symbol={args.symbol}, force={args.force}): {value}")
    else:
        # 计算所有特征
        print(f"计算所有特征 (date={args.date}, symbol={args.symbol}, force={args.force}):")
        print("-" * 60)
        results = compute_all_features(args.date, args.symbol, force=args.force)
        for feature_name, value in results.items():
            print(f"  {feature_name:20s}: {value}")


if __name__ == "__main__":
    main()

