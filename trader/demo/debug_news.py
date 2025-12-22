"""
调试脚本：查看数据库中的新闻数据格式
"""
import sys
import sqlite3
from pathlib import Path

# 添加项目根目录到 Python 路径
project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from trader.config import DB_PATH

def debug_news():
    """查看第一条新闻的原始数据"""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    
    cursor.execute("""
        SELECT id, stock_code, news
        FROM raw_data
        WHERE news IS NOT NULL AND news != ''
        ORDER BY id ASC
        LIMIT 1
    """)
    
    row = cursor.fetchone()
    
    if not row:
        print("未找到包含新闻的记录")
        conn.close()
        return
    
    print("=" * 80)
    print(f"记录 ID: {row['id']}")
    print(f"股票代码: {row['stock_code']}")
    print("=" * 80)
    print("\n原始新闻数据（前1000字符）:")
    print("-" * 80)
    news_raw = row['news']
    print(news_raw[:1000])
    print("-" * 80)
    print(f"\n总长度: {len(news_raw)} 字符")
    print(f"数据类型: {type(news_raw)}")
    print(f"是否以 {{ 开头: {news_raw.strip().startswith('{')}")
    print(f"是否以 [ 开头: {news_raw.strip().startswith('[')}")
    print(f"是否以 ' 开头: {news_raw.strip().startswith(\"'\")}")
    print(f"是否以 \" 开头: {news_raw.strip().startswith('\"')}")
    
    conn.close()

if __name__ == "__main__":
    debug_news()

