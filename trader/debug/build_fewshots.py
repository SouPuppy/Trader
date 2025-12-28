"""
Debug 工具：从数据库中获取数据构建 few-shot 样本
用于填充 trader/rag/fewshots.json 文件
"""
import sys
import sqlite3
import json
import re
from pathlib import Path
from typing import List, Dict, Optional
from datetime import datetime, timedelta
import pandas as pd

# 添加项目根目录到 Python 路径
project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from trader.config import DB_PATH
from trader.logger import get_logger
from trader.news.prepare import parse_news_json, clean_html

logger = get_logger(__name__)

# Few-shots 文件路径
FEWSHOTS_FILE = project_root / "trader" / "rag" / "fewshots.json"


def get_stock_codes(limit: int = 20) -> List[str]:
    """
    获取数据库中所有股票代码
    
    Args:
        limit: 返回前 N 支股票，默认 20
        
    Returns:
        股票代码列表
    """
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        query = """
            SELECT DISTINCT stock_code
            FROM raw_data
            WHERE stock_code IS NOT NULL
            ORDER BY stock_code ASC
            LIMIT ?
        """
        
        cursor.execute(query, (limit,))
        stock_codes = [row[0] for row in cursor.fetchall()]
        conn.close()
        
        logger.info(f"找到 {len(stock_codes)} 支股票")
        return stock_codes
        
    except Exception as e:
        logger.error(f"获取股票代码失败: {e}", exc_info=True)
        return []


def calculate_volatility(row: Dict) -> float:
    """
    计算单日波动率
    
    Args:
        row: 包含价格数据的字典
        
    Returns:
        波动率（百分比）
    """
    prev_close = row.get('prev_close')
    close_price = row.get('close_price')
    
    if prev_close is None or close_price is None or prev_close == 0:
        return 0.0
    
    # 计算价格变化百分比（绝对值）
    volatility = abs((close_price - prev_close) / prev_close) * 100
    
    return volatility


def get_news_for_date_range(stock_code: str, target_date: str, days: int = 7) -> List[Dict]:
    """
    获取指定日期前 N 天的新闻
    
    Args:
        stock_code: 股票代码
        target_date: 目标日期（格式：YYYY-MM-DD）
        days: 回看天数，默认 7 天
        
    Returns:
        新闻列表
    """
    try:
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        # 计算日期范围
        target_dt = datetime.strptime(target_date, "%Y-%m-%d")
        start_dt = target_dt - timedelta(days=days)
        start_date = start_dt.strftime("%Y-%m-%d")
        
        query = """
            SELECT 
                datetime,
                news
            FROM raw_data
            WHERE stock_code = ?
              AND datetime >= ?
              AND datetime <= ?
              AND news IS NOT NULL
              AND news != ''
            ORDER BY datetime ASC
        """
        
        cursor.execute(query, (stock_code, start_date, target_date))
        rows = cursor.fetchall()
        
        news_list = []
        for row in rows:
            news_raw = row['news']
            news_obj = parse_news_json(news_raw)
            
            if news_obj:
                # 清理 HTML 内容
                content = news_obj.get('content', '')
                if content:
                    content = clean_html(content)
                    # 清理多余的空白字符
                    content = ' '.join(content.split())
                
                news_item = {
                    'datetime': row['datetime'],
                    'publish_time': news_obj.get('publish_time', ''),
                    'title': news_obj.get('title', ''),
                    'content': content
                }
                news_list.append(news_item)
        
        conn.close()
        return news_list
        
    except Exception as e:
        logger.error(f"获取新闻失败: {e}", exc_info=True)
        return []


def get_stock_data_with_volatility(stock_code: str, top_n: int = 3) -> List[Dict]:
    """
    获取股票数据并计算波动率，返回波动最大的 N 天
    
    Args:
        stock_code: 股票代码
        top_n: 返回波动最大的 N 天，默认 3
        
    Returns:
        包含波动率数据的字典列表
    """
    try:
        conn = sqlite3.connect(DB_PATH)
        
        query = """
            SELECT 
                datetime,
                stock_code,
                prev_close,
                open_price,
                high_price,
                low_price,
                close_price,
                volume
            FROM raw_data
            WHERE stock_code = ?
              AND prev_close IS NOT NULL
              AND close_price IS NOT NULL
              AND prev_close > 0
            ORDER BY datetime DESC
        """
        
        df = pd.read_sql_query(query, conn, params=(stock_code,))
        conn.close()
        
        if df.empty:
            return []
        
        # 计算波动率和价格变化
        df['volatility'] = abs((df['close_price'] - df['prev_close']) / df['prev_close']) * 100
        df['price_change_pct'] = ((df['close_price'] - df['prev_close']) / df['prev_close']) * 100
        
        # 按波动率排序，取前 N 个
        df_sorted = df.nlargest(top_n, 'volatility')
        
        results = []
        for _, row in df_sorted.iterrows():
            result = {
                'datetime': row['datetime'],
                'stock_code': row['stock_code'],
                'prev_close': float(row['prev_close']),
                'close_price': float(row['close_price']),
                'volume': float(row['volume']) if pd.notna(row['volume']) else 0.0,
                'volatility': float(row['volatility']),
                'price_change_pct': float(row['price_change_pct'])
            }
            results.append(result)
        
        return results
        
    except Exception as e:
        logger.error(f"获取股票数据失败: {e}", exc_info=True)
        return []


def format_news_context(news_list: List[Dict]) -> str:
    """
    格式化新闻列表为上下文字符串
    
    Args:
        news_list: 新闻列表
        
    Returns:
        格式化的新闻上下文字符串
    """
    news_parts = []
    for news in news_list:
        datetime_str = news.get('datetime', '')
        publish_time = news.get('publish_time', '')
        title = news.get('title', '')
        content = news.get('content', '')
        
        news_text = f"[{datetime_str}]"
        if publish_time:
            news_text += f" {publish_time}"
        news_text += f"\n标题: {title}\n内容: {content}"
        news_parts.append(news_text)
    
    return "\n\n---\n\n".join(news_parts)


def build_few_shot_examples(
    num_stocks: int = 20,
    top_days_per_stock: int = 3,
    news_days: int = 7,
    max_examples: Optional[int] = None
) -> List[Dict]:
    """
    从数据库构建 few-shot 样本
    
    Args:
        num_stocks: 分析的股票数量，默认 20
        top_days_per_stock: 每支股票取波动最大的 N 天，默认 3
        news_days: 获取前 N 天的新闻，默认 7
        max_examples: 最大样本数量，如果为 None 则返回所有样本
        
    Returns:
        Few-shot 样本列表
    """
    logger.info(f"开始构建 few-shot 样本...")
    logger.info(f"参数: num_stocks={num_stocks}, top_days_per_stock={top_days_per_stock}, news_days={news_days}")
    
    stock_codes = get_stock_codes(limit=num_stocks)
    if not stock_codes:
        logger.error("未找到股票代码")
        return []
    
    examples = []
    
    for stock_code in stock_codes:
        logger.info(f"处理股票: {stock_code}")
        
        # 获取波动最大的几天
        volatile_days = get_stock_data_with_volatility(stock_code, top_n=top_days_per_stock)
        
        for day_data in volatile_days:
            # 如果设置了最大样本数，达到后停止
            if max_examples is not None and len(examples) >= max_examples:
                break
            
            date = day_data['datetime']
            
            # 获取前 N 天的新闻
            news_list = get_news_for_date_range(stock_code, date, days=news_days)
            
            # 格式化新闻上下文
            news_context = format_news_context(news_list)
            
            # 构建 few-shot example
            example = {
                "stock_code": stock_code,
                "date": date,
                "volatility": day_data['volatility'],
                "price_change_pct": day_data['price_change_pct'],
                "prev_close": day_data['prev_close'],
                "close_price": day_data['close_price'],
                "volume": day_data['volume'],
                "news_count": len(news_list),
                "news_context": news_context,
                "news_list": news_list
            }
            
            examples.append(example)
            logger.info(f"  添加样本: {date}, 波动率={day_data['volatility']:.2f}%, 新闻数={len(news_list)}")
        
        # 如果设置了最大样本数，达到后停止外层循环
        if max_examples is not None and len(examples) >= max_examples:
            break
    
    logger.info(f"共构建 {len(examples)} 个 few-shot 样本")
    return examples


def save_fewshots(examples: List[Dict], output_file: Path = None):
    """
    保存 few-shot 样本到 JSON 文件
    
    Args:
        examples: Few-shot 样本列表
        output_file: 输出文件路径，如果为 None 则使用默认路径
    """
    if output_file is None:
        output_file = FEWSHOTS_FILE
    
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(examples, f, indent=2, ensure_ascii=False)
    
    logger.info(f"Few-shot 样本已保存到: {output_file}")
    logger.info(f"共 {len(examples)} 个样本")


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="从数据库构建 few-shot 样本")
    parser.add_argument(
        "--num-stocks",
        type=int,
        default=20,
        help="分析的股票数量（默认: 20）"
    )
    parser.add_argument(
        "--top-days",
        type=int,
        default=3,
        help="每支股票取波动最大的 N 天（默认: 3）"
    )
    parser.add_argument(
        "--news-days",
        type=int,
        default=7,
        help="获取前 N 天的新闻（默认: 7）"
    )
    parser.add_argument(
        "--max-examples",
        type=int,
        default=None,
        help="最大样本数量（默认: 无限制）"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="输出文件路径（默认: trader/rag/fewshots.json）"
    )
    
    args = parser.parse_args()
    
    # 构建 few-shot 样本
    examples = build_few_shot_examples(
        num_stocks=args.num_stocks,
        top_days_per_stock=args.top_days,
        news_days=args.news_days,
        max_examples=args.max_examples
    )
    
    if not examples:
        logger.warning("未生成任何样本")
        return
    
    # 保存到文件
    output_file = Path(args.output) if args.output else None
    save_fewshots(examples, output_file)
    
    # 打印统计信息
    logger.info("")
    logger.info("=" * 60)
    logger.info("统计信息:")
    logger.info(f"  总样本数: {len(examples)}")
    
    # 按股票统计
    stock_counts = {}
    for ex in examples:
        stock = ex['stock_code']
        stock_counts[stock] = stock_counts.get(stock, 0) + 1
    
    logger.info(f"  涉及股票数: {len(stock_counts)}")
    logger.info(f"  平均波动率: {sum(ex['volatility'] for ex in examples) / len(examples):.2f}%")
    logger.info(f"  平均新闻数: {sum(ex['news_count'] for ex in examples) / len(examples):.1f}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()

