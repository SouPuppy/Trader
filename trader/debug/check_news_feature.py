"""
调试脚本：检查 news_count 特征为什么是 0
"""
import sqlite3
import pandas as pd
from pathlib import Path
import sys

# 添加项目根目录到 Python 路径
project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from trader.config import DB_PATH
from trader.logger import get_logger
from trader.features.features import load_news_data_for_dates

logger = get_logger(__name__)


def check_raw_data_news():
    """检查 raw_data 表中的新闻数据"""
    print("\n" + "=" * 80)
    print("1. 检查 raw_data 表中的新闻数据")
    print("=" * 80)
    
    conn = sqlite3.connect(DB_PATH)
    
    # 统计总记录数
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM raw_data")
    total = cursor.fetchone()[0]
    print(f"总记录数: {total}")
    
    # 统计有新闻的记录数
    cursor.execute("SELECT COUNT(*) FROM raw_data WHERE news IS NOT NULL AND news != ''")
    with_news = cursor.fetchone()[0]
    print(f"有新闻的记录数: {with_news}")
    
    # 获取一个样本
    cursor.execute("""
        SELECT datetime, stock_code, news
        FROM raw_data
        WHERE news IS NOT NULL AND news != ''
        LIMIT 1
    """)
    sample = cursor.fetchone()
    if sample:
        print(f"\n样本记录:")
        print(f"  datetime: {sample[0]}")
        print(f"  stock_code: {sample[1]}")
        print(f"  news (前200字符): {sample[2][:200] if sample[2] else 'None'}...")
    
    conn.close()


def check_news_data_table():
    """检查 news_data 表中的数据"""
    print("\n" + "=" * 80)
    print("2. 检查 news_data 表中的数据")
    print("=" * 80)
    
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # 检查表是否存在
    cursor.execute("""
        SELECT name FROM sqlite_master 
        WHERE type='table' AND name='news_data'
    """)
    table_exists = cursor.fetchone() is not None
    
    if not table_exists:
        print("news_data 表不存在！")
        conn.close()
        return
    
    print("news_data 表存在")
    
    # 统计记录数
    cursor.execute("SELECT COUNT(*) FROM news_data")
    count = cursor.fetchone()[0]
    print(f"总记录数: {count}")
    
    if count > 0:
        # 获取一个样本（使用列名明确指定顺序）
        cursor.execute("SELECT id, uuid, paraphrase, sentiment, impact, publish_date FROM news_data LIMIT 1")
        sample = cursor.fetchone()
        if sample:
            print(f"\n样本记录:")
            print(f"  id: {sample[0]}")
            print(f"  uuid: {sample[1]}")
            print(f"  paraphrase: {sample[2][:100] if sample[2] else 'None'}...")
            print(f"  sentiment: {sample[3]}")
            print(f"  impact: {sample[4]}")
            print(f"  publish_date: {sample[5]}")
    
    conn.close()


def test_load_news_data():
    """测试 load_news_data_for_dates 函数"""
    print("\n" + "=" * 80)
    print("3. 测试 load_news_data_for_dates 函数")
    print("=" * 80)
    
    # 获取一个股票代码
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT DISTINCT stock_code FROM raw_data LIMIT 1")
    row = cursor.fetchone()
    if not row:
        print("没有找到股票代码")
        conn.close()
        return
    
    symbol = row[0]
    print(f"测试股票代码: {symbol}")
    
    # 获取该股票的一些日期
    cursor.execute("""
        SELECT DISTINCT datetime
        FROM raw_data
        WHERE stock_code = ?
        ORDER BY datetime DESC
        LIMIT 5
    """, (symbol,))
    dates_rows = cursor.fetchall()
    dates = [row[0] for row in dates_rows]
    print(f"测试日期: {dates}")
    
    conn.close()
    
    # 测试加载新闻数据
    print(f"\n调用 load_news_data_for_dates(symbol='{symbol}', dates={dates})...")
    news_df = load_news_data_for_dates(symbol, dates)
    
    print(f"返回的 DataFrame:")
    print(f"  形状: {news_df.shape}")
    print(f"  列: {news_df.columns.tolist()}")
    if not news_df.empty:
        print(f"\n所有数据:")
        print(news_df)
        print(f"\n按日期分组统计:")
        if 'datetime' in news_df.columns:
            grouped = news_df.groupby(news_df['datetime'].dt.date).size()
            for date, count in grouped.items():
                print(f"  {date}: {count} 条新闻")
    else:
        print("  DataFrame 为空")


def test_news_count_feature():
    """测试 news_count 特征计算"""
    print("\n" + "=" * 80)
    print("4. 测试 news_count 特征计算")
    print("=" * 80)
    
    from trader.features import get_feature
    from trader.features.features import load_stock_data
    
    # 获取一个股票代码
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT DISTINCT stock_code FROM raw_data LIMIT 1")
    row = cursor.fetchone()
    if not row:
        print("没有找到股票代码")
        conn.close()
        return
    
    symbol = row[0]
    print(f"测试股票代码: {symbol}")
    
    conn.close()
    
    # 加载股票数据
    print(f"\n加载股票数据...")
    stock_df = load_stock_data(symbol, lookback=0)
    if stock_df.empty:
        print("股票数据为空")
        return
    
    print(f"股票数据形状: {stock_df.shape}")
    print(f"股票数据列: {stock_df.columns.tolist()}")
    print(f"\n股票数据前3行:")
    print(stock_df.head(3))
    
    # 获取 news_count 特征
    feature_spec = get_feature('news_count')
    if not feature_spec:
        print("找不到 news_count 特征")
        return
    
    print(f"\n特征规范:")
    print(f"  name: {feature_spec.name}")
    print(f"  lookback: {feature_spec.lookback}")
    
    # 计算特征
    print(f"\n计算特征...")
    try:
        # 先测试 load_news_data_for_dates 看看能加载到什么
        dates_list = [pd.Timestamp(row['datetime']).strftime('%Y-%m-%d') for _, row in stock_df.iterrows()]
        print(f"准备查询的日期: {dates_list}")
        news_df = load_news_data_for_dates(symbol, dates_list)
        print(f"加载到的新闻数据:")
        if not news_df.empty:
            print(news_df)
            if 'datetime' in news_df.columns:
                print(f"\n新闻数据的日期范围:")
                print(f"  最早: {news_df['datetime'].min()}")
                print(f"  最晚: {news_df['datetime'].max()}")
                print(f"\n按日期分组:")
                grouped = news_df.groupby(news_df['datetime'].dt.date).size()
                for date, count in grouped.items():
                    print(f"  {date}: {count} 条")
        else:
            print("  无新闻数据")
        
        result = feature_spec.compute(stock_df)
        print(f"\n特征计算结果:")
        print(f"结果类型: {type(result)}")
        print(f"结果形状: {result.shape}")
        print(f"结果索引: {result.index.tolist()}")
        print(f"\n结果值:")
        for idx, val in result.items():
            date_val = stock_df.loc[idx, 'datetime'] if 'datetime' in stock_df.columns else idx
            print(f"  索引 {idx} (日期: {date_val}): {val}")
        print(f"\n非空值数量: {result.notna().sum()}")
        print(f"非零值数量: {(result != 0).sum()}")
    except Exception as e:
        print(f"计算特征时出错: {e}")
        import traceback
        traceback.print_exc()


def main():
    """主函数"""
    print("=" * 80)
    print("调试 news_count 特征")
    print("=" * 80)
    
    if not DB_PATH.exists():
        print(f"数据库文件不存在: {DB_PATH}")
        return
    
    check_raw_data_news()
    check_news_data_table()
    test_load_news_data()
    test_news_count_feature()
    
    print("\n" + "=" * 80)
    print("调试完成")
    print("=" * 80)


if __name__ == "__main__":
    main()

