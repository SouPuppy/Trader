import csv
import sqlite3
import sys
from pathlib import Path

# 添加项目根目录到 Python 路径
project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from trader.config import DB_PATH, CSV_PATH
from trader.logger import get_logger

logger = get_logger(__name__)

def init_database():
    """
    读取 data.csv 并存入 data.sqlite3，创建相应的 schema
    如果数据库已存在且有数据，则跳过初始化
    """
    logger.info(f"检查数据库: {DB_PATH}")
    
    # 检查数据库是否已存在且有数据
    if DB_PATH.exists():
        try:
            conn = sqlite3.connect(DB_PATH)
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM raw_data")
            count = cursor.fetchone()[0]
            conn.close()
            
            if count > 0:
                logger.warning(
                    f"数据库已存在且包含 {count} 条记录，跳过初始化。"
                    f"如需重新初始化，请先删除数据库文件: {DB_PATH}"
                )
                return
            else:
                logger.info("数据库存在但为空，将继续初始化...")
        except sqlite3.Error as e:
            logger.warning(f"检查数据库时出错: {e}，将继续初始化...")
    
    # 检查 CSV 文件是否存在
    if not CSV_PATH.exists():
        logger.error(f"CSV 文件不存在: {CSV_PATH}")
        raise FileNotFoundError(f"CSV 文件不存在: {CSV_PATH}")
    
    logger.info(f"开始初始化数据库: {DB_PATH}")
    logger.info(f"读取 CSV 文件: {CSV_PATH}")
    
    # 连接数据库（如果不存在会自动创建）
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # 创建表 schema（使用英文列名）
    create_table_sql = """
    CREATE TABLE IF NOT EXISTS raw_data (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        datetime TEXT,
        stock_code TEXT,
        prev_close REAL,
        open_price REAL,
        high_price REAL,
        low_price REAL,
        close_price REAL,
        volume REAL,
        news TEXT,
        pe_ratio REAL,
        pe_ratio_ttm REAL,
        pcf_ratio_ttm REAL,
        pb_ratio REAL,
        ps_ratio REAL,
        ps_ratio_ttm REAL,
        analyzed INTEGER DEFAULT 0
    )
    """
    
    logger.debug("创建数据库表 schema")
    cursor.execute(create_table_sql)
    logger.debug("数据库表创建成功")
    
    # 检查并添加/迁移字段（如果不存在）
    cursor.execute("PRAGMA table_info(raw_data)")
    columns = [col[1] for col in cursor.fetchall()]
    
    # 检查并添加 datetime 字段（如果不存在）
    if 'datetime' not in columns:
        logger.info("添加 datetime 字段到 raw_data 表...")
        cursor.execute("ALTER TABLE raw_data ADD COLUMN datetime TEXT")
        conn.commit()
        logger.info("datetime 字段添加成功")
    
    # 检查并添加 analyzed 字段（如果不存在）
    if 'analyzed' not in columns:
        logger.info("添加 analyzed 字段到 raw_data 表...")
        cursor.execute("ALTER TABLE raw_data ADD COLUMN analyzed INTEGER DEFAULT 0")
        conn.commit()
        logger.info("analyzed 字段添加成功")
    else:
        # 如果 analyzed 字段存在但是 TEXT 类型，需要迁移为 INTEGER
        cursor.execute("PRAGMA table_info(raw_data)")
        column_info = cursor.fetchall()
        analyzed_col = next((col for col in column_info if col[1] == 'analyzed'), None)
        if analyzed_col and analyzed_col[2] != 'INTEGER':
            logger.info("检测到 analyzed 字段不是 INTEGER 类型，正在迁移...")
            # SQLite 不支持直接修改列类型，需要创建新表并迁移数据
            # 为了简单，我们只更新现有数据：将空字符串或 'v1' 改为 1，其他改为 0
            cursor.execute("UPDATE raw_data SET analyzed = 1 WHERE analyzed = '1' OR analyzed = 1 OR analyzed = 'v1'")
            cursor.execute("UPDATE raw_data SET analyzed = 0 WHERE analyzed = '0' OR analyzed = 0 OR analyzed = '' OR analyzed IS NULL")
            conn.commit()
            logger.info("analyzed 字段数据迁移完成")
    
    # 添加单条新闻分析结果字段（每条 raw_data 记录对应一条新闻）
    news_fields = [
        ('news_sentiment', 'INTEGER'),  # 单条新闻的情绪 [-10, 10]
        ('news_impact', 'INTEGER'),     # 单条新闻的影响强度 [0, 10]
    ]
    
    for field_name, field_type in news_fields:
        if field_name not in columns:
            logger.info(f"添加 {field_name} 字段到 raw_data 表...")
            cursor.execute(f"ALTER TABLE raw_data ADD COLUMN {field_name} {field_type}")
            conn.commit()
            logger.info(f"{field_name} 字段添加成功")
    
    # 注意：每条 raw_data 记录对应一条新闻，分析结果存储在 news_sentiment 和 news_impact 字段中
    # 计算特征时，通过 SQL 聚合这些字段来计算 mean/sum/count 等特征
    
    # 如果表已存在且有数据，可以选择清空表（可选）
    # cursor.execute("DELETE FROM data")
    
    # 读取 CSV 文件并插入数据
    # 增加 CSV 字段大小限制以处理长文本（新闻列可能很长）
    # 设置为 10MB 应该足够
    csv.field_size_limit(10 * 1024 * 1024)
    logger.debug("CSV 字段大小限制已设置为 10MB")
    
    with open(CSV_PATH, 'r', encoding='utf-8') as f:
        csv_reader = csv.reader(f)
        
        # 跳过表头
        header = next(csv_reader)
        logger.debug(f"CSV 表头: {header}")
        
        # 准备插入语句（使用英文列名）
        insert_sql = """
        INSERT INTO raw_data (
            datetime, stock_code, prev_close, open_price, high_price, low_price, close_price,
            volume, news, pe_ratio, pe_ratio_ttm, pcf_ratio_ttm, pb_ratio, ps_ratio, ps_ratio_ttm, analyzed
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """
        
        # 批量插入数据
        rows_to_insert = []
        processed_rows = 0
        skipped_rows = 0
        
        logger.info("开始读取 CSV 数据...")
        for row in csv_reader:
            # 跳过空行
            if not row or len(row) < 15:
                skipped_rows += 1
                continue
            
            processed_rows += 1
            if processed_rows % 1000 == 0:
                logger.debug(f"已处理 {processed_rows} 行数据...")
            
            # 处理数据：将空字符串转换为 None，数值字符串转换为 float
            def convert_value(val):
                if val == '' or val is None:
                    return None
                try:
                    return float(val)
                except ValueError:
                    return val
            
            # 提取数据
            # CSV 列顺序：空列(日期), thscode, 昨日收盘价, 开盘价, 最高价, 最低价, 收盘价, 成交量, 新闻, 市盈率, ...
            data = (
                row[0] if len(row) > 0 else None,  # datetime (CSV 第一列，日期)
                row[1] if len(row) > 1 else None,  # stock_code
                convert_value(row[2]) if len(row) > 2 else None,  # prev_close
                convert_value(row[3]) if len(row) > 3 else None,  # open_price
                convert_value(row[4]) if len(row) > 4 else None,  # high_price
                convert_value(row[5]) if len(row) > 5 else None,  # low_price
                convert_value(row[6]) if len(row) > 6 else None,  # close_price
                convert_value(row[7]) if len(row) > 7 else None,  # volume
                row[8] if len(row) > 8 else None,  # news
                convert_value(row[9]) if len(row) > 9 else None,  # pe_ratio
                convert_value(row[10]) if len(row) > 10 else None,  # pe_ratio_ttm
                convert_value(row[11]) if len(row) > 11 else None,  # pcf_ratio_ttm
                convert_value(row[12]) if len(row) > 12 else None,  # pb_ratio
                convert_value(row[13]) if len(row) > 13 else None,  # ps_ratio
                convert_value(row[14]) if len(row) > 14 else None,  # ps_ratio_ttm
                0,  # analyzed (默认未分析)
            )
            
            rows_to_insert.append(data)
        
        if skipped_rows > 0:
            logger.warning(f"跳过了 {skipped_rows} 行无效数据")
        
        logger.info(f"准备插入 {len(rows_to_insert)} 条记录到数据库...")
        # 批量插入
        cursor.executemany(insert_sql, rows_to_insert)
        logger.info(f"批量插入完成，共 {len(rows_to_insert)} 条记录")
    
    # 提交事务
    conn.commit()
    logger.info("数据库事务已提交")
    
    # 显示插入的行数
    cursor.execute("SELECT COUNT(*) FROM raw_data")
    count = cursor.fetchone()[0]
    logger.info(f"成功插入 {count} 条记录到数据库")
    
    # 关闭连接
    conn.close()
    logger.info(f"数据库已创建: {DB_PATH}")

if __name__ == "__main__":
    # 建议使用 cmd 目录下的脚本：python trader/cmd/db_init.py
    logger.info("正在执行数据库初始化...")
    init_database()

