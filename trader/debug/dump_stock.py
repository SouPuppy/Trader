"""
从 SQLite 数据库中读取并输出股票数据的完整信息
支持根据 schema 显示所有字段，包括 analyzed 状态和 news_data 分析结果
"""
import sys
import sqlite3
import json
import argparse
from pathlib import Path

# 添加项目根目录到 Python 路径
project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from trader.config import DB_PATH
from trader.logger import get_logger
from trader.news.prepare import parse_news_json, clean_html, convert_publish_time_to_utc

logger = get_logger(__name__)


def format_value(value, field_type='REAL'):
    """格式化字段值"""
    if value is None:
        return 'N/A'
    if field_type == 'REAL' and isinstance(value, (int, float)):
        return f"{value:,.2f}" if abs(value) >= 1000 else f"{value:.2f}"
    return str(value)


def get_row_value(row, key, default=None):
    """安全地从 sqlite3.Row 对象获取值"""
    try:
        value = row[key]
        return value if value is not None else default
    except (KeyError, IndexError):
        return default


def dump_stock(record_id: int = None, stock_code: str = None, show_full: bool = False):
    """
    从数据库中读取并输出股票数据
    
    Args:
        record_id: 指定记录 ID（可选）
        stock_code: 指定股票代码（可选）
        show_full: 是否显示完整内容（默认只显示预览）
    """
    logger.info(f"连接数据库: {DB_PATH}")
    
    # 检查数据库是否存在
    if not DB_PATH.exists():
        logger.error(f"数据库文件不存在: {DB_PATH}")
        logger.info("请先运行 ./script/init_db.sh 初始化数据库")
        return
    
    try:
        # 连接数据库
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        # 构建查询条件
        if record_id:
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
                    ps_ratio_ttm,
                    analyzed
                FROM raw_data
                WHERE id = ?
            """, (record_id,))
        elif stock_code:
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
                    ps_ratio_ttm,
                    analyzed
                FROM raw_data
                WHERE stock_code = ?
                ORDER BY id ASC
                LIMIT 1
            """, (stock_code,))
        else:
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
                    ps_ratio_ttm,
                    analyzed
                FROM raw_data
                ORDER BY id ASC
                LIMIT 1
            """)
        
        row = cursor.fetchone()
        
        if row is None:
            logger.warning("数据库中没有数据")
            conn.close()
            return
        
        logger.info("成功读取数据")
        
        # 输出完整数据
        print("\n" + "=" * 80)
        print("股票数据（完整信息）")
        print("=" * 80)
        
        # 基本信息
        print("\n【基本信息】")
        print("-" * 80)
        print(f"ID:                    {row['id']}")
        print(f"日期 (datetime):        {get_row_value(row, 'datetime', 'N/A')}")
        print(f"股票代码 (stock_code):  {row['stock_code']}")
        analyzed_value = get_row_value(row, 'analyzed', 0)
        print(f"分析状态 (analyzed):    {'已分析' if analyzed_value else '未分析'}")
        
        # 价格数据
        print("\n【价格数据】")
        print("-" * 80)
        print(f"昨日收盘价 (prev_close): {format_value(row['prev_close'])}")
        print(f"开盘价 (open_price):    {format_value(row['open_price'])}")
        print(f"最高价 (high_price):    {format_value(row['high_price'])}")
        print(f"最低价 (low_price):     {format_value(row['low_price'])}")
        print(f"收盘价 (close_price):   {format_value(row['close_price'])}")
        
        # 交易数据
        print("\n【交易数据】")
        print("-" * 80)
        volume = row['volume']
        if volume:
            print(f"成交量 (volume):        {volume:,.0f} 股")
        else:
            print(f"成交量 (volume):        N/A")
        
        # 财务指标
        print("\n【财务指标】")
        print("-" * 80)
        print(f"市盈率 (pe_ratio):      {format_value(row['pe_ratio'])}")
        print(f"市盈率TTM (pe_ratio_ttm): {format_value(row['pe_ratio_ttm'])}")
        print(f"市现率TTM (pcf_ratio_ttm): {format_value(row['pcf_ratio_ttm'])}")
        print(f"市净率 (pb_ratio):      {format_value(row['pb_ratio'])}")
        print(f"市销率 (ps_ratio):      {format_value(row['ps_ratio'])}")
        print(f"市销率TTM (ps_ratio_ttm): {format_value(row['ps_ratio_ttm'])}")
        
        # 新闻数据
        print("\n【新闻数据】")
        print("-" * 80)
        if row['news']:
            news_obj = parse_news_json(row['news'])
            if news_obj:
                publish_time = news_obj.get('publish_time', 'N/A')
                title = news_obj.get('title', '')
                content = news_obj.get('content', '')
                
                print(f"发布时间: {publish_time}")
                
                if title:
                    cleaned_title = clean_html(title)
                    title_display = cleaned_title if show_full else cleaned_title[:200]
                    print(f"标题: {title_display}")
                    if len(cleaned_title) > 200 and not show_full:
                        print(f"  ... (还有 {len(cleaned_title) - 200} 字符)")
                
                if content:
                    cleaned_content = clean_html(content)
                    content_display = cleaned_content if show_full else cleaned_content[:500]
                    print(f"\n内容预览:")
                    print(content_display)
                    if len(cleaned_content) > 500 and not show_full:
                        print(f"\n... (还有 {len(cleaned_content) - 500} 字符)")
            else:
                # 解析失败，显示原始数据预览
                news_preview = row['news'][:500] + "..." if len(row['news']) > 500 else row['news']
                print("⚠ 无法解析新闻 JSON，显示原始数据:")
                print(news_preview)
        else:
            print("无新闻数据")
        
        # 分析结果（如果已分析）
        analyzed = get_row_value(row, 'analyzed', 0)
        if analyzed:
            print("\n【分析结果】")
            print("-" * 80)
            try:
                # 尝试解析新闻获取 publish_date
                news_obj = parse_news_json(row['news'])
                if news_obj:
                    publish_time_str = news_obj.get('publish_time', '')
                    if publish_time_str:
                        publish_date = convert_publish_time_to_utc(publish_time_str)
                        if publish_date:
                            cursor.execute("""
                                SELECT id, uuid, paraphrase, sentiment, impact, publish_date
                                FROM news_data
                                WHERE publish_date = ?
                                ORDER BY id DESC
                                LIMIT 1
                            """, (publish_date,))
                            analysis_row = cursor.fetchone()
                            if analysis_row:
                                print(f"UUID: {analysis_row['uuid']}")
                                print(f"发布时间 (UTC): {analysis_row['publish_date']}")
                                print(f"情绪方向: {analysis_row['sentiment']} (范围: -10 到 10)")
                                print(f"影响强度: {analysis_row['impact']} (范围: 0 到 10)")
                                paraphrase = analysis_row['paraphrase'] or ''
                                paraphrase_display = paraphrase if show_full else paraphrase[:300]
                                print(f"主题/摘要: {paraphrase_display}")
                                if len(paraphrase) > 300 and not show_full:
                                    print(f"  ... (还有 {len(paraphrase) - 300} 字符)")
                            else:
                                print("⚠ 未找到匹配的分析结果")
                        else:
                            print("⚠ 无法转换 publish_time 为 UTC")
                    else:
                        print("⚠ 新闻中没有 publish_time 字段")
                else:
                    print("⚠ 无法解析新闻 JSON，无法查询分析结果")
            except Exception as e:
                logger.debug(f"查询分析结果时出错: {e}")
                print(f"⚠ 查询分析结果时出错: {e}")
        
        print("\n" + "=" * 80)
        
        # 输出为字典格式（便于程序使用）
        if show_full:
            print("\n【JSON 格式】")
            print("-" * 80)
            data_dict = {
                'id': row['id'],
                'datetime': get_row_value(row, 'datetime'),
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
                'analyzed': bool(get_row_value(row, 'analyzed', 0)),
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


def main():
    """主函数，支持命令行参数"""
    parser = argparse.ArgumentParser(description='从数据库读取并输出股票数据')
    parser.add_argument('--id', type=int, help='指定记录 ID')
    parser.add_argument('--stock', type=str, help='指定股票代码')
    parser.add_argument('--full', action='store_true', help='显示完整内容（包括 JSON 格式）')
    
    args = parser.parse_args()
    
    dump_stock(record_id=args.id, stock_code=args.stock, show_full=args.full)


if __name__ == "__main__":
    main()

