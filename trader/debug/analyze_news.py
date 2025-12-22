"""
演示：分析新闻
从数据库读取第一条新闻并使用 DeepSeek API 进行分析（仅演示，不保存结果）
"""
import sys
import json
import sqlite3
from pathlib import Path

# 添加项目根目录到 Python 路径
project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from trader.config import DB_PATH
from trader.news.analyze import analyze
from trader.news.prepare import parse_news_json
from trader.logger import get_logger

logger = get_logger(__name__)


def main():
    """分析数据库中的第一条新闻（仅演示，不保存结果）"""
    logger.info("开始分析新闻（演示模式，不保存结果）...")
    
    try:
        # 连接数据库
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        # 查询第一条有新闻的记录
        cursor.execute("""
            SELECT id, stock_code, news
            FROM raw_data
            WHERE news IS NOT NULL AND news != ''
            ORDER BY id ASC
            LIMIT 1
        """)
        
        row = cursor.fetchone()
        
        if not row:
            logger.warning("未找到包含新闻的记录")
            conn.close()
            return
        
        raw_data_id = row['id']
        stock_code = row['stock_code']
        news_raw = row['news']
        
        logger.info(f"找到新闻记录: ID={raw_data_id}, stock_code={stock_code}")
        
        # 解析新闻 JSON
        news_obj = parse_news_json(news_raw)
        
        if not news_obj:
            logger.error(f"无法解析新闻 JSON。原始数据: {news_raw[:200] if news_raw else 'None'}")
            conn.close()
            return
        
        # 构建新闻字典用于分析
        news_dict = {
            'publish_time': news_obj.get('publish_time', ''),
            'title': news_obj.get('title', ''),
            'content': news_obj.get('content', '')
        }
        
        if not news_dict['title'] and not news_dict['content']:
            logger.error("新闻内容为空")
            conn.close()
            return
        
        # 进行分析
        logger.info("开始分析新闻...")
        analysis_result = analyze(news_dict)
        
        conn.close()
        
        if not analysis_result:
            logger.error("分析失败")
            return
        
        # 打印分析结果
        print("\n" + "=" * 80)
        print("新闻分析结果（演示模式）")
        print("=" * 80)
        print(f"股票代码: {stock_code}")
        print(f"记录 ID: {raw_data_id}")
        print(f"发布时间: {news_dict['publish_time']}")
        print(f"标题: {news_dict['title'][:100]}...")
        print("\n分析结果:")
        print(f"  主题/摘要: {analysis_result['paraphrase']}")
        print(f"  情绪方向: {analysis_result['sentiment']} (范围: -10 到 10)")
        print(f"  影响强度: {analysis_result['impact']} (范围: 0 到 10)")
        print("=" * 80)
        
        # 输出 JSON 格式
        print("\nJSON 格式:")
        print(json.dumps(analysis_result, indent=2, ensure_ascii=False))
        print("\n注意: 这是演示模式，分析结果未保存到数据库")
        print("要保存结果，请使用 trader/news/prepare.py --process")
        
    except Exception as e:
        logger.error(f"分析新闻失败: {e}", exc_info=True)


if __name__ == "__main__":
    main()

