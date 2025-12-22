"""
新闻准备模块
用于清理 HTML、提取文本、解析新闻 JSON 等操作
以及统计未分析的新闻数量和处理未分析的新闻
"""
import sys
import json
import re
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Optional
from html import unescape
from html.parser import HTMLParser

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

from trader.config import DB_PATH
from trader.logger import get_logger

logger = get_logger(__name__)


class HTMLTextExtractor(HTMLParser):
    """HTML 文本提取器"""
    def __init__(self):
        super().__init__()
        self.text = []
        self.skip_tags = {'script', 'style', 'meta', 'link', 'head'}
        self.current_tag = None
    
    def handle_starttag(self, tag, attrs):
        self.current_tag = tag.lower()
        if tag.lower() in {'br', 'p', 'div'}:
            self.text.append('\n')
    
    def handle_endtag(self, tag):
        if tag.lower() in {'p', 'div', 'li'}:
            self.text.append('\n')
        self.current_tag = None
    
    def handle_data(self, data):
        if self.current_tag not in self.skip_tags:
            self.text.append(data)
    
    def get_text(self):
        text = ''.join(self.text)
        # 清理多余的空白字符
        text = re.sub(r'\n\s*\n', '\n\n', text)
        text = re.sub(r'[ \t]+', ' ', text)
        return text.strip()


def clean_html(html_content: str) -> str:
    """
    清理 HTML 内容，提取纯文本
    
    Args:
        html_content: HTML 格式的字符串
        
    Returns:
        清理后的纯文本
    """
    if not html_content:
        return ""
    
    try:
        # HTML 解码
        decoded = unescape(html_content)
        
        # 提取文本
        extractor = HTMLTextExtractor()
        extractor.feed(decoded)
        text = extractor.get_text()
        
        return text
    except Exception as e:
        logger.warning(f"HTML 清理失败: {e}")
        # 如果解析失败，尝试简单的正则替换
        text = re.sub(r'<[^>]+>', '', html_content)
        text = unescape(text)
        return text.strip()


def parse_news_json(news_str: str) -> Optional[Dict]:
    """
    解析新闻 JSON 字符串
    
    Args:
        news_str: JSON 格式的新闻字符串（可能是单个对象、列表或字符串）
        
    Returns:
        解析后的新闻字典，如果解析失败返回 None
    """
    if not news_str:
        return None
    
    # 去除首尾空白
    news_str = news_str.strip()
    
    # 如果字符串为空，返回 None
    if not news_str:
        return None
    
    try:
        # 尝试直接解析 JSON
        news_obj = json.loads(news_str)
        
        # 如果是列表，取第一个元素
        if isinstance(news_obj, list):
            if len(news_obj) > 0:
                news_obj = news_obj[0]
            else:
                logger.warning("新闻列表为空")
                return None
        
        # 确保是字典类型
        if not isinstance(news_obj, dict):
            logger.warning(f"新闻数据不是字典类型: {type(news_obj)}")
            return None
        
        return news_obj
        
    except json.JSONDecodeError as e:
        # 尝试修复常见的 JSON 格式问题
        logger.warning(f"JSON 解析失败: {e}")
        logger.debug(f"原始数据前200字符: {news_str[:200]}")
        
        # 尝试修复单引号问题（Python 字典格式）
        try:
            # 如果看起来像 Python 字典格式（使用单引号），尝试用 ast.literal_eval
            import ast
            if news_str.startswith("{") or news_str.startswith("["):
                news_obj = ast.literal_eval(news_str)
                if isinstance(news_obj, list) and len(news_obj) > 0:
                    news_obj = news_obj[0]
                if isinstance(news_obj, dict):
                    logger.info("使用 ast.literal_eval 成功解析 Python 字典格式")
                    return news_obj
        except Exception as ast_e:
            logger.debug(f"ast.literal_eval 也失败: {ast_e}")
        
        return None
        
    except Exception as e:
        logger.warning(f"解析新闻数据失败: {e}", exc_info=True)
        return None


def extract_text_from_news(news_data: str) -> str:
    """
    从新闻数据中提取文本内容
    
    Args:
        news_data: JSON 格式的新闻数据字符串
        
    Returns:
        提取的文本内容
    """
    news_obj = parse_news_json(news_data)
    
    if not news_obj:
        return ""
    
    # 提取标题和内容
    title = news_obj.get('title', '')
    content = news_obj.get('content', '')
    
    # 清理 HTML
    title_text = clean_html(title) if title else ""
    content_text = clean_html(content) if content else ""
    
    # 组合标题和内容
    if title_text and content_text:
        return f"{title_text}\n\n{content_text}"
    elif title_text:
        return title_text
    elif content_text:
        return content_text
    else:
        return ""


def get_news_summary(news_data: str) -> Dict:
    """
    获取新闻摘要信息
    
    Args:
        news_data: JSON 格式的新闻数据字符串
        
    Returns:
        包含标题、发布时间等信息的字典
    """
    news_obj = parse_news_json(news_data)
    
    if not news_obj:
        return {
            'title': '',
            'publish_time': '',
            'content_preview': ''
        }
    
    title = news_obj.get('title', '')
    publish_time = news_obj.get('publish_time', '')
    content = news_obj.get('content', '')
    
    # 提取内容预览（前 200 字符）
    content_preview = clean_html(content)[:200] if content else ""
    
    return {
        'title': title,
        'publish_time': publish_time,
        'content_preview': content_preview
    }


def ensure_news_data_table():
    """确保 news_data 表存在并具有正确的 schema"""
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        # 创建 news_data 表
        create_table_sql = """
        CREATE TABLE IF NOT EXISTS news_data (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            uuid TEXT UNIQUE NOT NULL,
            paraphrase TEXT,
            sentiment INTEGER CHECK(sentiment >= -10 AND sentiment <= 10),
            impact INTEGER CHECK(impact >= 0 AND impact <= 10),
            publish_date TEXT
        )
        """
        
        cursor.execute(create_table_sql)
        conn.commit()
        logger.debug("news_data 表已确保存在")
        
        conn.close()
    except Exception as e:
        logger.error(f"确保 news_data 表失败: {e}", exc_info=True)
        raise


def ensure_analyzed_column():
    """确保 raw_data 表有 analyzed 字段（INTEGER 类型，0 表示未分析，1 表示已分析）"""
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        # 检查 analyzed 字段是否存在
        cursor.execute("PRAGMA table_info(raw_data)")
        columns = [col[1] for col in cursor.fetchall()]
        
        if 'analyzed' not in columns:
            logger.info("添加 analyzed 字段到 raw_data 表...")
            cursor.execute("ALTER TABLE raw_data ADD COLUMN analyzed INTEGER DEFAULT 0")
            conn.commit()
            logger.info("analyzed 字段添加成功")
        else:
            # 如果字段存在但是 TEXT 类型，需要迁移数据
            cursor.execute("PRAGMA table_info(raw_data)")
            column_info = cursor.fetchall()
            analyzed_col = next((col for col in column_info if col[1] == 'analyzed'), None)
            if analyzed_col and analyzed_col[2] != 'INTEGER':
                logger.info("检测到 analyzed 字段不是 INTEGER 类型，正在迁移...")
                # SQLite 不支持直接修改列类型，需要创建新表
                # 但为了简单，我们只更新现有数据：将空字符串或 'v1' 改为 1，其他改为 0
                cursor.execute("UPDATE raw_data SET analyzed = 1 WHERE analyzed = '1' OR analyzed = 1 OR analyzed = 'v1'")
                cursor.execute("UPDATE raw_data SET analyzed = 0 WHERE analyzed = '0' OR analyzed = 0 OR analyzed = '' OR analyzed IS NULL")
                conn.commit()
                logger.info("analyzed 字段数据迁移完成")
            logger.debug("analyzed 字段已存在")
        
        conn.close()
    except Exception as e:
        logger.error(f"确保 analyzed 字段失败: {e}", exc_info=True)
        raise


def convert_publish_time_to_utc(publish_time_str: str) -> Optional[str]:
    """
    将发布时间字符串转换为 UTC 格式
    
    Args:
        publish_time_str: 发布时间字符串
        
    Returns:
        UTC 格式的时间字符串，格式: "YYYY-MM-DD HH:MM:SS UTC"，如果转换失败返回 None
    """
    if not publish_time_str:
        return None
    
    try:
        # 尝试解析时间字符串（支持多种格式）
        time_formats = [
            '%Y-%m-%d %H:%M:%S',
            '%Y-%m-%d %H:%M:%S.%f',
            '%Y-%m-%dT%H:%M:%S',
            '%Y-%m-%dT%H:%M:%S.%f',
            '%Y-%m-%dT%H:%M:%SZ',
            '%Y-%m-%dT%H:%M:%S.%fZ',
        ]
        
        dt = None
        for fmt in time_formats:
            try:
                dt = datetime.strptime(publish_time_str, fmt)
                break
            except ValueError:
                continue
        
        if dt:
            # 如果时间没有时区信息，假设为 UTC
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            else:
                # 转换为 UTC
                dt = dt.astimezone(timezone.utc)
            
            # 格式化为 UTC 字符串：YYYY-MM-DD HH:MM:SS UTC
            return dt.strftime('%Y-%m-%d %H:%M:%S UTC')
        else:
            logger.warning(f"无法解析时间格式: {publish_time_str}")
            return None
    except Exception as e:
        logger.warning(f"时间转换失败: {e}, 原始时间: {publish_time_str}")
        return None


def count_unanalyzed_news() -> Dict[str, int]:
    """
    统计 raw_data 表中未分析的新闻数量
    
    Returns:
        包含统计信息的字典：
        - total: 总记录数
        - with_news: 有新闻内容的记录数
        - unanalyzed: 未分析的记录数（analyzed = 0 或 NULL）
        - analyzed: 已分析的记录数（analyzed = 1）
    """
    try:
        ensure_analyzed_column()
        
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        # 统计总记录数
        cursor.execute("SELECT COUNT(*) FROM raw_data")
        total = cursor.fetchone()[0]
        
        # 统计有新闻内容的记录数
        cursor.execute("SELECT COUNT(*) FROM raw_data WHERE news IS NOT NULL AND news != ''")
        with_news = cursor.fetchone()[0]
        
        # 统计未分析的记录数（analyzed = 0 或 NULL）
        cursor.execute("SELECT COUNT(*) FROM raw_data WHERE (analyzed IS NULL OR analyzed = 0) AND news IS NOT NULL AND news != ''")
        unanalyzed = cursor.fetchone()[0]
        
        # 统计已分析的记录数（analyzed = 1）
        cursor.execute("SELECT COUNT(*) FROM raw_data WHERE analyzed = 1 AND news IS NOT NULL AND news != ''")
        analyzed = cursor.fetchone()[0]
        
        conn.close()
        
        stats = {
            'total': total,
            'with_news': with_news,
            'unanalyzed': unanalyzed,
            'analyzed': analyzed
        }
        
        logger.info(f"新闻统计: 总计 {total} 条记录，有新闻 {with_news} 条，未分析 {unanalyzed} 条，已分析 {analyzed} 条")
        
        return stats
        
    except Exception as e:
        logger.error(f"统计未分析新闻失败: {e}", exc_info=True)
        raise


def process_unanalyzed_news(limit: int = 1) -> Dict[str, int]:
    """
    处理未分析的新闻：分析并保存到 news_data 表
    
    流程：
    1. 找到 analyzed = false 的记录
    2. 如果该记录在 news_data 中有记录（通过 publish_date 匹配），先删除并 warning（防止部分数据）
    3. 使用 analyze 处理新闻
    4. 处理完后存入 news_data 数据库
    5. 最后把 analyzed 设置为 true（保证原子操作）
    
    Args:
        limit: 每次处理的记录数，默认为 1
        
    Returns:
        包含处理结果的字典：
        - processed: 成功处理的记录数
        - failed: 处理失败的记录数
        - skipped: 跳过的记录数（无新闻内容或解析失败）
    """
    try:
        from trader.news.analyze import analyze
        
        ensure_analyzed_column()
        ensure_news_data_table()
        
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        # 查询未分析的新闻记录
        cursor.execute("""
            SELECT id, stock_code, news
            FROM raw_data
            WHERE (analyzed = 0 OR analyzed IS NULL)
              AND news IS NOT NULL 
              AND news != ''
            ORDER BY id ASC
            LIMIT ?
        """, (limit,))
        
        rows = cursor.fetchall()
        
        if not rows:
            logger.info("没有未分析的新闻记录")
            conn.close()
            return {'processed': 0, 'failed': 0, 'skipped': 0}
        
        logger.info(f"找到 {len(rows)} 条未分析的新闻记录，开始处理...")
        
        processed = 0
        failed = 0
        skipped = 0
        
        # 使用 tqdm 显示处理进度
        for row in tqdm(rows, desc="处理新闻", total=len(rows), unit="条"):
            raw_data_id = row['id']
            stock_code = row['stock_code']
            news_raw = row['news']
            
            try:
                # 解析新闻 JSON
                news_obj = parse_news_json(news_raw)
                
                if not news_obj:
                    logger.warning(f"记录 ID {raw_data_id}: 无法解析新闻 JSON，跳过")
                    skipped += 1
                    continue
                
                # 构建新闻字典用于分析
                news_dict = {
                    'publish_time': news_obj.get('publish_time', ''),
                    'title': news_obj.get('title', ''),
                    'content': news_obj.get('content', '')
                }
                
                if not news_dict['title'] and not news_dict['content']:
                    logger.warning(f"记录 ID {raw_data_id}: 新闻内容为空，跳过")
                    skipped += 1
                    continue
                
                # 提取 publish_time 并转换为 UTC
                publish_time_str = news_dict.get('publish_time', '')
                publish_date = convert_publish_time_to_utc(publish_time_str) if publish_time_str else None
                
                # 进行分析（在事务外进行，因为这是外部 API 调用，可能耗时较长）
                logger.info(f"记录 ID {raw_data_id} (stock_code={stock_code}): 开始分析新闻...")
                analysis_result = analyze(news_dict)
                
                if not analysis_result:
                    logger.error(f"记录 ID {raw_data_id}: 分析失败")
                    failed += 1
                    continue
                
                # 开始显式事务（保证原子操作：删除旧记录、插入新记录、更新 analyzed 字段）
                # SQLite 默认 autocommit，我们需要显式管理事务
                try:
                    # 检查并删除 news_data 中可能存在的部分记录（通过 publish_date 匹配）
                    if publish_date:
                        cursor.execute("""
                            SELECT id, uuid FROM news_data 
                            WHERE publish_date = ?
                        """, (publish_date,))
                        
                        existing_records = cursor.fetchall()
                        if existing_records:
                            logger.warning(
                                f"记录 ID {raw_data_id}: 发现 news_data 表中存在部分数据 "
                                f"（publish_date={publish_date}），共 {len(existing_records)} 条记录，正在删除..."
                            )
                            for record in existing_records:
                                cursor.execute("DELETE FROM news_data WHERE id = ?", (record['id'],))
                                logger.warning(f"已删除 news_data 记录 ID {record['id']}, UUID {record['uuid']}")
                    
                    # 生成 UUID
                    news_uuid = str(uuid.uuid4())
                    
                    # 插入到 news_data 表
                    insert_sql = """
                    INSERT INTO news_data (uuid, paraphrase, sentiment, impact, publish_date)
                    VALUES (?, ?, ?, ?, ?)
                    """
                    
                    cursor.execute(insert_sql, (
                        news_uuid,
                        analysis_result.get('paraphrase', ''),
                        analysis_result.get('sentiment', 0),
                        analysis_result.get('impact', 0),
                        publish_date
                    ))
                    
                    # 更新 raw_data 表的 analyzed 字段为 1
                    update_sql = "UPDATE raw_data SET analyzed = 1 WHERE id = ?"
                    cursor.execute(update_sql, (raw_data_id,))
                    
                    # 提交事务（保证原子操作）
                    conn.commit()
                    
                    processed += 1
                    logger.info(
                        f"记录 ID {raw_data_id}: 分析完成并保存成功 "
                        f"(UUID={news_uuid}, sentiment={analysis_result.get('sentiment')}, "
                        f"impact={analysis_result.get('impact')})"
                    )
                    
                except Exception as e:
                    # 如果处理失败，回滚当前记录的事务
                    conn.rollback()
                    logger.error(f"记录 ID {raw_data_id}: 数据库操作失败: {e}", exc_info=True)
                    failed += 1
                    continue
                
            except Exception as e:
                # 如果处理失败，回滚当前记录的事务
                conn.rollback()
                logger.error(f"记录 ID {raw_data_id}: 处理失败: {e}", exc_info=True)
                failed += 1
        
        conn.close()
        
        result = {
            'processed': processed,
            'failed': failed,
            'skipped': skipped
        }
        
        logger.info(
            f"处理完成: 成功 {processed} 条，失败 {failed} 条，跳过 {skipped} 条"
        )
        
        return result
        
    except Exception as e:
        logger.error(f"处理未分析新闻失败: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    # 建议使用 cmd 目录下的脚本：python trader/cmd/prepare_news.py
    import argparse
    
    parser = argparse.ArgumentParser(description='新闻准备和处理工具')
    parser.add_argument('--limit', type=int, default=None, help='处理的记录数限制（默认：处理全部）')
    parser.add_argument('--stats-only', action='store_true', help='仅显示统计信息，不处理')
    
    args = parser.parse_args()
    
    # 确保 analyzed 字段存在
    ensure_analyzed_column()
    
    # 统计未分析的新闻
    stats = count_unanalyzed_news()
    
    print("\n" + "=" * 60)
    print("新闻分析统计")
    print("=" * 60)
    print(f"总记录数:        {stats['total']}")
    print(f"有新闻内容:      {stats['with_news']}")
    print(f"未分析:          {stats['unanalyzed']}")
    print(f"已分析:          {stats['analyzed']}")
    print("=" * 60)
    
    # 默认处理全部未分析的新闻，除非指定了 --stats-only
    if not args.stats_only:
        if stats['unanalyzed'] > 0:
            limit_text = f"{args.limit} 条" if args.limit else "全部"
            print(f"\n开始处理 {limit_text} 未分析的新闻...")
            
            # 如果没有指定 limit，处理全部（使用一个很大的数字）
            limit = args.limit if args.limit else stats['unanalyzed']
            result = process_unanalyzed_news(limit=limit)
            
            print("\n" + "=" * 60)
            print("处理结果")
            print("=" * 60)
            print(f"成功处理:        {result['processed']}")
            print(f"处理失败:        {result['failed']}")
            print(f"跳过:            {result['skipped']}")
            print("=" * 60)
        else:
            print("\n没有未分析的新闻需要处理")
    else:
        print("\n仅显示统计信息，未进行处理")

