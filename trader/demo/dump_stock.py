"""
从 SQLite 数据库中读取并输出第一条股票数据的完整信息
"""
import sys
import sqlite3
import json
from pathlib import Path

# 添加项目根目录到 Python 路径
project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from trader.config import DB_PATH
from trader.logger import get_logger

logger = get_logger(__name__)

def dump_first_stock():
    """从数据库中读取并输出第一条完整数据"""
    
    logger.info(f"连接数据库: {DB_PATH}")
    
    # 检查数据库是否存在
    if not DB_PATH.exists():
        logger.error(f"数据库文件不存在: {DB_PATH}")
        logger.info("请先运行 ./script/init_db.sh 初始化数据库")
        return
    
    try:
        # 连接数据库
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row  # 使用 Row 对象，可以通过列名访问
        cursor = conn.cursor()
        
        # 查询第一条数据
        cursor.execute("""
            SELECT 
                id,
                datetime,
                stock_code,
                prev_close,
                open_price,
                high_price,
                low_price,
                close_price,
                volume,
                news,
                pe_ratio,
                pe_ratio_ttm,
                pcf_ratio_ttm,
                pb_ratio,
                ps_ratio,
                ps_ratio_ttm
            FROM raw_data
            ORDER BY id ASC
            LIMIT 1
        """)
        
        row = cursor.fetchone()
        
        if row is None:
            logger.warning("数据库中没有数据")
            conn.close()
            return
        
        logger.info("成功读取第一条数据")
        logger.info("=" * 80)
        
        # 输出完整数据
        print("\n第一条股票数据（完整信息）:")
        print("=" * 80)
        print(f"ID:                    {row['id']}")
        print(f"日期 (datetime):        {row.get('datetime', 'N/A')}")
        print(f"股票代码 (stock_code):  {row['stock_code']}")
        print(f"昨日收盘价 (prev_close): {row['prev_close']}")
        print(f"开盘价 (open_price):    {row['open_price']}")
        print(f"最高价 (high_price):    {row['high_price']}")
        print(f"最低价 (low_price):     {row['low_price']}")
        print(f"收盘价 (close_price):   {row['close_price']}")
        print(f"成交量 (volume):        {row['volume']:,.0f}")
        print(f"市盈率 (pe_ratio):      {row['pe_ratio']}")
        print(f"市盈率TTM (pe_ratio_ttm): {row['pe_ratio_ttm']}")
        print(f"市现率TTM (pcf_ratio_ttm): {row['pcf_ratio_ttm']}")
        print(f"市净率 (pb_ratio):      {row['pb_ratio']}")
        print(f"市销率 (ps_ratio):      {row['ps_ratio']}")
        print(f"市销率TTM (ps_ratio_ttm): {row['ps_ratio_ttm']}")
        
        # 处理新闻数据（JSON格式）
        print("\n新闻 (news):")
        print("-" * 80)
        if row['news']:
            try:
                news_data = json.loads(row['news'])
                if isinstance(news_data, list) and len(news_data) > 0:
                    # 如果是列表，显示第一条新闻
                    first_news = news_data[0]
                    print(f"发布时间: {first_news.get('publish_time', 'N/A')}")
                    print(f"标题: {first_news.get('title', 'N/A')}")
                    content = first_news.get('content', '')
                    if content:
                        # 只显示前200个字符
                        content_preview = content[:200] + "..." if len(content) > 200 else content
                        print(f"内容预览: {content_preview}")
                else:
                    print(news_data)
            except json.JSONDecodeError:
                # 如果不是有效的JSON，直接显示原始内容的前500个字符
                news_preview = row['news'][:500] + "..." if len(row['news']) > 500 else row['news']
                print(news_preview)
        else:
            print("无新闻数据")
        
        print("=" * 80)
        
        # 输出为字典格式（便于程序使用）
        logger.info("\n数据字典格式:")
        data_dict = {
            'id': row['id'],
            'datetime': row.get('datetime'),
            'stock_code': row['stock_code'],
            'prev_close': row['prev_close'],
            'open_price': row['open_price'],
            'high_price': row['high_price'],
            'low_price': row['low_price'],
            'close_price': row['close_price'],
            'volume': row['volume'],
            'pe_ratio': row['pe_ratio'],
            'pe_ratio_ttm': row['pe_ratio_ttm'],
            'pcf_ratio_ttm': row['pcf_ratio_ttm'],
            'pb_ratio': row['pb_ratio'],
            'ps_ratio': row['ps_ratio'],
            'ps_ratio_ttm': row['ps_ratio_ttm'],
            'news': row['news']
        }
        print(json.dumps(data_dict, indent=2, ensure_ascii=False))
        
        conn.close()
        logger.info("数据库连接已关闭")
        
    except sqlite3.Error as e:
        logger.error(f"数据库操作错误: {e}")
        raise
    except Exception as e:
        logger.error(f"发生错误: {e}", exc_info=True)
        raise

if __name__ == "__main__":
    dump_first_stock()

