"""
特征缓存模块
提供特征值的数据库缓存功能，避免重复计算
使用 wide format: (stock_code, datetime) 作为主键，每个特征作为一列
"""
import sqlite3
from typing import Optional, Dict, List
from datetime import datetime as dt
import pandas as pd
from trader.config import DB_PATH
from trader.logger import get_logger

logger = get_logger(__name__)


def get_all_feature_names() -> List[str]:
    """
    获取所有已注册的特征名称
    """
    try:
        from trader.features import get_feature_names
        return get_feature_names()
    except Exception as e:
        logger.warning(f"获取特征名称失败: {e}")
        return []


def ensure_features_table():
    """
    确保 features 表存在，如果不存在则创建（wide format）
    表结构: (stock_code, datetime) 作为主键，每个特征作为一列
    """
    if not DB_PATH.exists():
        logger.warning(f"数据库文件不存在: {DB_PATH}")
        return False
    
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        # 检查表是否存在
        cursor.execute("""
            SELECT name FROM sqlite_master 
            WHERE type='table' AND name='features'
        """)
        table_exists = cursor.fetchone() is not None
        
        if not table_exists:
            # 获取所有特征名称
            feature_names = get_all_feature_names()
            if not feature_names:
                logger.warning("无法获取特征名称，跳过创建 features 表")
                conn.close()
                return False
            
            # 构建 CREATE TABLE 语句
            # 主键: (stock_code, datetime)
            # 每个特征作为一列: feature_name REAL
            feature_columns = ',\n            '.join([f'"{name}" REAL' for name in feature_names])
            
            create_table_sql = f"""
            CREATE TABLE features (
                stock_code TEXT NOT NULL,
                datetime TEXT NOT NULL,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                {feature_columns},
                PRIMARY KEY (stock_code, datetime)
            )
            """
            
            cursor.execute(create_table_sql)
            logger.info(f"创建 features 表（wide format），包含 {len(feature_names)} 个特征列")
        else:
            # 表已存在，检查是否需要添加新特征列
            cursor.execute("PRAGMA table_info(features)")
            existing_columns = {row[1] for row in cursor.fetchall()}
            
            feature_names = get_all_feature_names()
            new_features = [name for name in feature_names if name not in existing_columns]
            
            if new_features:
                logger.info(f"发现 {len(new_features)} 个新特征，添加到 features 表")
                for feature_name in new_features:
                    try:
                        cursor.execute(f'ALTER TABLE features ADD COLUMN "{feature_name}" REAL')
                        logger.debug(f"添加特征列: {feature_name}")
                    except sqlite3.OperationalError as e:
                        # 列可能已存在（并发情况）
                        logger.debug(f"特征列 {feature_name} 可能已存在: {e}")
        
        # 创建索引以提高查询性能
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_features_stock_date 
            ON features(stock_code, datetime)
        """)
        
        conn.commit()
        conn.close()
        
        logger.debug("features 表已确保存在")
        return True
        
    except Exception as e:
        logger.error(f"创建 features 表时出错: {e}", exc_info=True)
        return False


def get_cached_feature(stock_code: str, datetime: str, feature_name: str) -> Optional[float]:
    """
    从缓存中获取特征值（wide format）
    
    Args:
        stock_code: 股票代码
        datetime: 日期字符串 (YYYY-MM-DD)
        feature_name: 特征名称
        
    Returns:
        特征值，如果不存在则返回 None
    """
    if not DB_PATH.exists():
        return None
    
    try:
        ensure_features_table()
        
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        # Wide format: 直接查询特征列
        query = f'SELECT "{feature_name}" FROM features WHERE stock_code = ? AND datetime = ?'
        
        cursor.execute(query, (stock_code, datetime))
        row = cursor.fetchone()
        
        conn.close()
        
        if row is not None and row[0] is not None:
            logger.debug(f"从缓存获取特征: {stock_code} {datetime} {feature_name} = {row[0]}")
            return float(row[0])
        
        return None
        
    except sqlite3.OperationalError as e:
        # 特征列可能不存在（新特征）
        logger.debug(f"特征列 {feature_name} 不存在: {e}")
        return None
    except Exception as e:
        logger.error(f"从缓存获取特征时出错: {e}", exc_info=True)
        return None


def cache_feature(stock_code: str, datetime: str, feature_name: str, feature_value: Optional[float]):
    """
    将特征值存储到缓存中（wide format）
    
    Args:
        stock_code: 股票代码
        datetime: 日期字符串 (YYYY-MM-DD)
        feature_name: 特征名称
        feature_value: 特征值（可以为 None）
    """
    if not DB_PATH.exists():
        logger.warning(f"数据库文件不存在，无法缓存特征: {DB_PATH}")
        return
    
    try:
        ensure_features_table()
        
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        # Wide format: 使用 INSERT OR REPLACE，只更新指定特征列
        # 首先检查记录是否存在
        cursor.execute("SELECT stock_code FROM features WHERE stock_code = ? AND datetime = ?", 
                      (stock_code, datetime))
        exists = cursor.fetchone() is not None
        
        if exists:
            # 更新现有记录的特征列
            query = f'UPDATE features SET "{feature_name}" = ? WHERE stock_code = ? AND datetime = ?'
            cursor.execute(query, (feature_value, stock_code, datetime))
        else:
            # 插入新记录，只设置这个特征值，其他特征为 NULL
            query = f'INSERT INTO features (stock_code, datetime, "{feature_name}") VALUES (?, ?, ?)'
            cursor.execute(query, (stock_code, datetime, feature_value))
        
        conn.commit()
        conn.close()
        
        logger.debug(f"缓存特征: {stock_code} {datetime} {feature_name} = {feature_value}")
        
    except sqlite3.OperationalError as e:
        # 特征列可能不存在，需要先添加
        logger.debug(f"特征列 {feature_name} 不存在，尝试添加: {e}")
        ensure_features_table()  # 重新确保表结构
        # 重试一次
        try:
            conn = sqlite3.connect(DB_PATH)
            cursor = conn.cursor()
            cursor.execute(f'UPDATE features SET "{feature_name}" = ? WHERE stock_code = ? AND datetime = ?',
                          (feature_value, stock_code, datetime))
            conn.commit()
            conn.close()
        except Exception as e2:
            logger.error(f"重试缓存特征时出错: {e2}", exc_info=True)
    except Exception as e:
        logger.error(f"缓存特征时出错: {e}", exc_info=True)


def cache_features_batch(features_data: List[tuple]):
    """
    批量缓存特征值（wide format）
    
    Args:
        features_data: 列表，每个元素为 (stock_code, datetime, feature_name, feature_value)
    """
    if not DB_PATH.exists() or not features_data:
        return
    
    try:
        ensure_features_table()
        
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        # 按 (stock_code, datetime) 分组组织数据
        from collections import defaultdict
        grouped = defaultdict(dict)
        for stock_code, datetime, feature_name, feature_value in features_data:
            key = (stock_code, datetime)
            grouped[key][feature_name] = feature_value
        
        # 批量插入/更新
        for (stock_code, datetime), features_dict in grouped.items():
            # 检查记录是否存在
            cursor.execute("SELECT stock_code FROM features WHERE stock_code = ? AND datetime = ?",
                          (stock_code, datetime))
            exists = cursor.fetchone() is not None
            
            if exists:
                # 更新多个特征列
                updates = []
                values = []
                for feature_name, feature_value in features_dict.items():
                    updates.append(f'"{feature_name}" = ?')
                    values.append(feature_value)
                values.extend([stock_code, datetime])
                
                query = f'UPDATE features SET {", ".join(updates)} WHERE stock_code = ? AND datetime = ?'
                cursor.execute(query, values)
            else:
                # 插入新记录
                feature_names = list(features_dict.keys())
                feature_values = [features_dict[name] for name in feature_names]
                columns = ', '.join([f'"{name}"' for name in feature_names])
                placeholders = ', '.join(['?'] * (len(feature_names) + 2))
                
                query = f'INSERT INTO features (stock_code, datetime, {columns}) VALUES ({placeholders})'
                cursor.execute(query, [stock_code, datetime] + feature_values)
        
        conn.commit()
        conn.close()
        
        logger.debug(f"批量缓存 {len(features_data)} 个特征值，涉及 {len(grouped)} 条记录")
        
    except Exception as e:
        logger.error(f"批量缓存特征时出错: {e}", exc_info=True)


def get_cached_features_batch(
    stock_code: str, 
    dates: List[str], 
    feature_names: List[str]
) -> Dict[str, Dict[str, Optional[float]]]:
    """
    批量从缓存中获取特征值（wide format）
    一次查询获取所有特征，性能更好
    
    Args:
        stock_code: 股票代码
        dates: 日期列表
        feature_names: 特征名称列表
        
    Returns:
        字典: {date: {feature_name: value}}
    """
    if not DB_PATH.exists() or not dates or not feature_names:
        return {}
    
    try:
        ensure_features_table()
        
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        # Wide format: 一次查询获取所有特征列
        dates_placeholder = ','.join(['?'] * len(dates))
        feature_columns = ', '.join([f'"{name}"' for name in feature_names])
        
        query = f"""
            SELECT datetime, {feature_columns}
            FROM features 
            WHERE stock_code = ? 
              AND datetime IN ({dates_placeholder})
        """
        
        cursor.execute(query, (stock_code, *dates))
        rows = cursor.fetchall()
        
        conn.close()
        
        # 组织结果
        result = {}
        for row in rows:
            datetime_str = row[0]
            result[datetime_str] = {}
            for i, feature_name in enumerate(feature_names, start=1):
                value = row[i] if i < len(row) else None
                result[datetime_str][feature_name] = float(value) if value is not None else None
        
        logger.debug(f"从缓存批量获取特征: {stock_code}, {len(dates)} 个日期, {len(feature_names)} 个特征, 找到 {len(result)} 个日期的数据")
        
        return result
        
    except sqlite3.OperationalError as e:
        # 某些特征列可能不存在
        logger.debug(f"查询特征时出错（可能某些特征列不存在）: {e}")
        return {}
    except Exception as e:
        logger.error(f"批量从缓存获取特征时出错: {e}", exc_info=True)
        return {}


def get_cached_all_features(stock_code: str, datetime: str) -> Dict[str, Optional[float]]:
    """
    从缓存中获取某个日期的所有特征值（wide format）
    一次查询获取所有特征，性能更好
    
    Args:
        stock_code: 股票代码
        datetime: 日期字符串 (YYYY-MM-DD)
        
    Returns:
        字典: {feature_name: value}
    """
    if not DB_PATH.exists():
        return {}
    
    try:
        ensure_features_table()
        
        feature_names = get_all_feature_names()
        if not feature_names:
            return {}
        
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        # Wide format: 一次查询获取所有特征列
        feature_columns = ', '.join([f'"{name}"' for name in feature_names])
        
        query = f"""
            SELECT {feature_columns}
            FROM features 
            WHERE stock_code = ? AND datetime = ?
        """
        
        cursor.execute(query, (stock_code, datetime))
        row = cursor.fetchone()
        
        conn.close()
        
        if row is None:
            return {}
        
        # 组织结果
        result = {}
        for i, feature_name in enumerate(feature_names):
            value = row[i] if i < len(row) else None
            result[feature_name] = float(value) if value is not None else None
        
        logger.debug(f"从缓存获取所有特征: {stock_code} {datetime}, 共 {len(result)} 个特征")
        
        return result
        
    except sqlite3.OperationalError as e:
        # 某些特征列可能不存在
        logger.debug(f"查询所有特征时出错（可能某些特征列不存在）: {e}")
        return {}
    except Exception as e:
        logger.error(f"从缓存获取所有特征时出错: {e}", exc_info=True)
        return {}


def clear_cache(stock_code: Optional[str] = None, feature_name: Optional[str] = None):
    """
    清除缓存（wide format）
    
    Args:
        stock_code: 如果指定，只清除该股票的缓存
        feature_name: 如果指定，只清除该特征列的值（设为 NULL）
    """
    if not DB_PATH.exists():
        return
    
    try:
        ensure_features_table()
        
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        if stock_code and feature_name:
            # 清除特定股票的特定特征值（设为 NULL）
            query = f'UPDATE features SET "{feature_name}" = NULL WHERE stock_code = ?'
            cursor.execute(query, (stock_code,))
        elif stock_code:
            # 删除特定股票的所有记录
            query = "DELETE FROM features WHERE stock_code = ?"
            cursor.execute(query, (stock_code,))
        elif feature_name:
            # 清除所有记录的特定特征值（设为 NULL）
            query = f'UPDATE features SET "{feature_name}" = NULL'
            cursor.execute(query)
        else:
            # 删除所有记录
            query = "DELETE FROM features"
            cursor.execute(query)
        
        conn.commit()
        conn.close()
        
        logger.info(f"清除缓存: stock_code={stock_code}, feature_name={feature_name}")
        
    except Exception as e:
        logger.error(f"清除缓存时出错: {e}", exc_info=True)

