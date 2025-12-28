"""
Debug 工具：找出20支股票波动最大的3个点，并输出前7天的所有新闻
"""
import sys
import sqlite3
import json
import re
import ast
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
from trader.news.prepare import parse_news_json

logger = get_logger(__name__)


def get_all_stock_codes(limit: int = 20) -> List[str]:
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


def load_stock_data_with_volatility(stock_code: str) -> pd.DataFrame:
    """
    加载股票数据并计算波动率
    
    Args:
        stock_code: 股票代码
        
    Returns:
        DataFrame 包含价格数据和波动率
    """
    try:
        conn = sqlite3.connect(DB_PATH)
        
        query = """
            SELECT 
                id,
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
            ORDER BY datetime ASC
        """
        
        df = pd.read_sql_query(query, conn, params=(stock_code,))
        conn.close()
        
        if df.empty:
            return df
        
        # 计算波动率
        df['volatility'] = df.apply(calculate_volatility, axis=1)
        
        return df
        
    except Exception as e:
        logger.error(f"加载股票 {stock_code} 数据失败: {e}", exc_info=True)
        return pd.DataFrame()


def get_top_volatile_days(df: pd.DataFrame, top_n: int = 3) -> pd.DataFrame:
    """
    获取波动最大的 N 天
    
    Args:
        df: 包含波动率的数据框
        top_n: 返回前 N 天
        
    Returns:
        波动最大的 N 天数据
    """
    if df.empty:
        return df
    
    # 按波动率降序排序
    top_days = df.nlargest(top_n, 'volatility')
    
    return top_days


def get_news_for_date_range(stock_code: str, end_date: str, days: int = 7) -> List[Dict]:
    """
    获取指定日期往前 N 天的所有新闻
    
    Args:
        stock_code: 股票代码
        end_date: 结束日期（格式: YYYY-MM-DD）
        days: 往前查询的天数，默认7天
        
    Returns:
        新闻列表，每个元素包含 datetime, news 等信息
    """
    try:
        # 解析结束日期
        end_dt = datetime.strptime(end_date, '%Y-%m-%d')
        start_dt = end_dt - timedelta(days=days - 1)  # 包含结束日期本身
        
        start_date_str = start_dt.strftime('%Y-%m-%d')
        
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        query = """
            SELECT datetime, news
            FROM raw_data
            WHERE stock_code = ? 
                AND datetime >= ? 
                AND datetime <= ?
                AND news IS NOT NULL 
                AND news != ''
            ORDER BY datetime ASC
        """
        
        cursor.execute(query, (stock_code, start_date_str, end_date))
        rows = cursor.fetchall()
        conn.close()
        
        news_list = []
        for row in rows:
            date_str, news_str = row
            if not news_str or not news_str.strip():
                continue
            
            # 直接解析 JSON，保留所有新闻项
            try:
                # 先尝试 ast.literal_eval（处理 Python 字典格式）
                if news_str.strip().startswith("{") or news_str.strip().startswith("["):
                    try:
                        news_obj = ast.literal_eval(news_str.strip())
                    except (ValueError, SyntaxError):
                        news_obj = json.loads(news_str.strip())
                else:
                    news_obj = json.loads(news_str.strip())
                
                # 如果是列表，展开所有新闻项
                if isinstance(news_obj, list):
                    for item in news_obj:
                        if isinstance(item, dict):
                            news_list.append({
                                'datetime': date_str,
                                'publish_time': item.get('publish_time', ''),
                                'title': item.get('title', ''),
                                'content': item.get('content', '')
                            })
                elif isinstance(news_obj, dict):
                    # 单个新闻对象
                    news_list.append({
                        'datetime': date_str,
                        'publish_time': news_obj.get('publish_time', ''),
                        'title': news_obj.get('title', ''),
                        'content': news_obj.get('content', '')
                    })
            except (json.JSONDecodeError, ValueError, SyntaxError) as e:
                logger.debug(f"解析新闻 JSON 失败 (日期: {date_str}): {e}")
                continue
        
        return news_list
        
    except Exception as e:
        logger.error(f"获取新闻失败: {e}", exc_info=True)
        return []


def print_stock_volatile_days(stock_code: str, top_days: pd.DataFrame):
    """
    打印股票的波动最大的天数及其前7天的新闻
    
    Args:
        stock_code: 股票代码
        top_days: 波动最大的天数数据框
    """
    print(f"\n{'='*100}")
    print(f"股票: {stock_code}")
    print(f"{'='*100}")
    
    if top_days.empty:
        print("  无数据")
        return
    
    for idx, (_, row) in enumerate(top_days.iterrows(), 1):
        print(f"\n【第 {idx} 名 - 波动率: {row['volatility']:.2f}%】")
        print(f"  日期: {row['datetime']}")
        print(f"  前收盘价: ${row['prev_close']:.2f}")
        print(f"  开盘价: ${row['open_price']:.2f}")
        print(f"  最高价: ${row['high_price']:.2f}")
        print(f"  最低价: ${row['low_price']:.2f}")
        print(f"  收盘价: ${row['close_price']:.2f}")
        print(f"  价格变化: ${row['close_price'] - row['prev_close']:.2f} ({((row['close_price'] - row['prev_close']) / row['prev_close'] * 100):+.2f}%)")
        print(f"  成交量: {row['volume']:,.0f}")
        
        # 获取前7天的新闻
        print(f"\n  【前7天新闻】")
        news_list = get_news_for_date_range(stock_code, str(row['datetime']), days=7)
        
        if not news_list:
            print("    无新闻数据")
        else:
            print(f"    共找到 {len(news_list)} 条新闻:")
            for news_idx, news in enumerate(news_list, 1):
                print(f"\n    新闻 {news_idx}:")
                print(f"      日期: {news['datetime']}")
                if news['publish_time']:
                    print(f"      发布时间: {news['publish_time']}")
                if news['title']:
                    print(f"      标题: {news['title']}")
                if news['content']:
                    # 移除HTML标签（简单处理）
                    content = re.sub(r'<[^>]+>', '', news['content'])
                    # 输出完整内容
                    print(f"      内容:")
                    # 按行输出，每行添加缩进
                    for line in content.split('\n'):
                        if line.strip():
                            print(f"        {line.strip()}")


def save_results(all_results: Dict[str, List[Dict]], output_file: Path):
    """
    保存结果到 JSON 文件
    
    Args:
        all_results: 所有股票的结果字典
        output_file: 输出文件路径
    """
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    
    logger.info(f"结果已保存到: {output_file}")


def format_as_few_shot_examples(all_results: Dict[str, List[Dict]], limit: Optional[int] = None) -> List[Dict]:
    """
    将结果整理成 few-shot 示例格式
    
    Args:
        all_results: 所有股票的结果字典
        limit: 限制返回的示例数量，如果为 None 则返回所有示例
        
    Returns:
        Few-shot 示例列表，每个示例包含：
        - stock_code: 股票代码
        - date: 日期
        - volatility: 波动率
        - price_change_pct: 价格变化百分比
        - news_context: 前7天的新闻上下文（合并的文本，已清理HTML）
        - news_list: 新闻列表
    """
    examples = []
    
    for stock_code, stock_data in all_results.items():
        for day_data in stock_data:
            # 如果设置了限制，达到限制后停止
            if limit is not None and len(examples) >= limit:
                break
            
            # 合并前7天的新闻为上下文
            news_context_parts = []
            for news in day_data.get('news', []):
                # 清理 HTML 标签
                content = news.get('content', '')
                if content:
                    content = re.sub(r'<[^>]+>', '', content)
                    # 清理多余的空白字符
                    content = ' '.join(content.split())
                
                title = news.get('title', '')
                publish_time = news.get('publish_time', '')
                news_date = news.get('datetime', '')
                
                news_text = f"[{news_date}]"
                if publish_time:
                    news_text += f" {publish_time}"
                news_text += f"\n标题: {title}\n内容: {content}"
                news_context_parts.append(news_text)
            
            news_context = "\n\n---\n\n".join(news_context_parts)
            
            example = {
                "stock_code": stock_code,
                "date": day_data.get('datetime', ''),
                "volatility": day_data.get('volatility', 0.0),
                "price_change_pct": day_data.get('price_change_pct', 0.0),
                "prev_close": day_data.get('prev_close', 0.0),
                "close_price": day_data.get('close_price', 0.0),
                "volume": day_data.get('volume', 0.0),
                "news_count": day_data.get('news_count', 0),
                "news_context": news_context,
                "news_list": day_data.get('news', [])
            }
            
            examples.append(example)
        
        # 如果设置了限制，达到限制后停止外层循环
        if limit is not None and len(examples) >= limit:
            break
    
    return examples


def save_few_shot_examples(examples: List[Dict], output_file: Path, format_type: str = "json"):
    """
    保存 few-shot 示例到文件
    
    Args:
        examples: Few-shot 示例列表
        output_file: 输出文件路径
        format_type: 格式类型，"json" 或 "txt"（用于直接作为 prompt）
    """
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    if format_type == "json":
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(examples, f, indent=2, ensure_ascii=False)
        logger.info(f"Few-shot 示例已保存到: {output_file}")
    
    elif format_type == "txt":
        # 格式化为文本 prompt 格式
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("# Few-Shot Examples: Stock Volatility Prediction\n\n")
            f.write("Based on news context, predict stock volatility.\n\n")
            f.write("=" * 100 + "\n\n")
            
            for idx, example in enumerate(examples, 1):
                f.write(f"## Example {idx}\n\n")
                f.write(f"**Stock:** {example['stock_code']}\n")
                f.write(f"**Date:** {example['date']}\n")
                f.write(f"**Volatility:** {example['volatility']:.2f}%\n")
                f.write(f"**Price Change:** {example['price_change_pct']:.2f}%\n")
                f.write(f"**Previous Close:** ${example['prev_close']:.2f}\n")
                f.write(f"**Close Price:** ${example['close_price']:.2f}\n")
                f.write(f"**Volume:** {example['volume']:,.0f}\n")
                f.write(f"**News Count:** {example['news_count']}\n\n")
                
                f.write("**News Context (Previous 7 Days):**\n")
                f.write("-" * 80 + "\n")
                f.write(example['news_context'])
                f.write("\n" + "-" * 80 + "\n\n")
                
                f.write("=" * 100 + "\n\n")
        
        logger.info(f"Few-shot 文本格式已保存到: {output_file}")
    
    elif format_type == "prompt":
        # 格式化为可以直接用于 LLM prompt 的格式
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("You are a financial analyst. Based on the news context, predict the stock volatility.\n\n")
            f.write("Examples:\n\n")
            
            for idx, example in enumerate(examples, 1):
                f.write(f"Example {idx}:\n")
                f.write(f"Stock: {example['stock_code']}\n")
                f.write(f"Date: {example['date']}\n")
                f.write(f"News Context:\n{example['news_context'][:2000]}...\n")  # 限制长度
                f.write(f"Actual Volatility: {example['volatility']:.2f}%\n")
                f.write(f"Actual Price Change: {example['price_change_pct']:.2f}%\n\n")
            
            f.write("\nNow, analyze the following news and predict the volatility:\n")
        
        logger.info(f"Few-shot prompt 格式已保存到: {output_file}")


def main():
    """主函数：找出20支股票波动最大的3个点，并输出前7天的所有新闻"""
    num_stocks = 20
    top_days = 3
    
    logger.info(f"开始分析 {num_stocks} 支股票，每支股票找出波动最大的 {top_days} 天，并输出前7天的新闻")
    
    # 获取所有股票代码
    stock_codes = get_all_stock_codes(limit=num_stocks)
    
    if not stock_codes:
        logger.error("未找到股票数据")
        return
    
    # 存储所有结果
    all_results = {}
    
    # 尝试导入 tqdm 用于进度显示
    try:
        from tqdm import tqdm
        use_tqdm = True
    except ImportError:
        use_tqdm = False
    
    iterator = tqdm(stock_codes, desc="分析股票", unit="支") if use_tqdm else stock_codes
    
    # 分析每支股票
    for stock_code in iterator:
        try:
            # 加载股票数据
            df = load_stock_data_with_volatility(stock_code)
            
            if df.empty:
                logger.warning(f"股票 {stock_code} 无数据，跳过")
                continue
            
            # 获取波动最大的天数
            top_days_df = get_top_volatile_days(df, top_n=top_days)
            
            if top_days_df.empty:
                continue
            
            # 打印结果（包含新闻）
            print_stock_volatile_days(stock_code, top_days_df)
            
            # 准备保存的数据
            stock_results = []
            for _, row in top_days_df.iterrows():
                # 获取前7天的新闻
                news_list = get_news_for_date_range(stock_code, str(row['datetime']), days=7)
                
                stock_results.append({
                    'datetime': str(row['datetime']),
                    'volatility': float(row['volatility']),
                    'prev_close': float(row['prev_close']),
                    'open_price': float(row['open_price']),
                    'high_price': float(row['high_price']),
                    'low_price': float(row['low_price']),
                    'close_price': float(row['close_price']),
                    'volume': float(row['volume']),
                    'price_change_pct': float((row['close_price'] - row['prev_close']) / row['prev_close'] * 100),
                    'news_count': len(news_list),
                    'news': news_list
                })
            
            all_results[stock_code] = stock_results
            
        except Exception as e:
            logger.error(f"分析股票 {stock_code} 时发生异常: {e}", exc_info=True)
            continue
    
    # 保存原始结果
    output_file = Path(f"output/debug/volatile_days_{num_stocks}stocks_{top_days}days.json")
    save_results(all_results, output_file)
    
    # 整理成 few-shot 格式（只选择 10 个示例）
    few_shot_limit = 10
    few_shot_examples = format_as_few_shot_examples(all_results, limit=few_shot_limit)
    
    # 保存 few-shot 示例（JSON 格式）
    few_shot_json_file = Path(f"output/debug/few_shot_examples_{few_shot_limit}examples.json")
    save_few_shot_examples(few_shot_examples, few_shot_json_file, format_type="json")
    
    # 保存 few-shot 示例（文本格式）
    few_shot_txt_file = Path(f"output/debug/few_shot_examples_{few_shot_limit}examples.txt")
    save_few_shot_examples(few_shot_examples, few_shot_txt_file, format_type="txt")
    
    # 保存 few-shot 示例（Prompt 格式）
    few_shot_prompt_file = Path(f"output/debug/few_shot_prompt_{few_shot_limit}examples.txt")
    save_few_shot_examples(few_shot_examples, few_shot_prompt_file, format_type="prompt")
    
    print(f"\n{'='*100}")
    print(f"分析完成！共分析 {len(all_results)} 支股票")
    print(f"原始结果已保存到: {output_file}")
    print(f"Few-shot 示例（共 {len(few_shot_examples)} 个）:")
    print(f"  - JSON 格式: {few_shot_json_file}")
    print(f"  - 文本格式: {few_shot_txt_file}")
    print(f"  - Prompt 格式: {few_shot_prompt_file}")
    print(f"{'='*100}")


if __name__ == "__main__":
    main()

