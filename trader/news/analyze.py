"""
新闻分析模块
使用 DeepSeek API 分析新闻内容，生成摘要、情绪和影响强度
"""
import sys
import json
import re
from pathlib import Path
from typing import Dict, Optional, Tuple, Union, List

# 添加项目根目录到 Python 路径
project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from trader.config import get_deepseek_api_key, DB_PATH
from trader.logger import get_logger
from trader.news.prepare import clean_html, parse_news_json

logger = get_logger(__name__)

PROMPT = """
You are a professional financial news analyst. Please analyze the following news content and provide analysis results according to the requirements.

News Content:
{content}

Please analyze according to the following requirements:

1. **paraphrase (News Summary/Topics)**: Summarize the core themes and key information of the news. Break down the main topics into at most 3 sub-topics, each separated by semicolons. Keep it concise and clear, not exceeding 150 words.
   Format: "Sub-topic 1; Sub-topic 2; Sub-topic 3" (use fewer if there are fewer topics)

2. **sentiment (Sentiment Direction)**: Evaluate the sentiment impact of the news on the stock market, ranging from -10 to 10.
   - 10: Extremely positive, major positive news
   - 5: Clearly positive
   - 0: Neutral, no significant impact
   - -5: Clearly negative
   - -10: Extremely negative, major negative news

3. **impact (Impact Intensity)**: Evaluate the intensity of the news impact on stock prices, ranging from 0 to 10.
   - 10: Extremely high impact, may cause significant stock price volatility
   - 7: High impact
   - 5: Moderate impact
   - 3: Low impact
   - 0: Almost no impact

Please output the results in JSON format as follows:
{{
    "paraphrase": "Sub-topic 1; Sub-topic 2; Sub-topic 3",
    "sentiment": sentiment_value (integer from -10 to 10),
    "impact": impact_value (integer from 0 to 10)
}}
"""


def preprocess_html(raw_html: str) -> str:
    """
    预处理 HTML：清理并提取纯文本
    
    Args:
        raw_html: 原始 HTML 内容
        
    Returns:
        清理后的文本内容
    """
    if not raw_html:
        return ""
    
    # 清理 HTML，提取纯文本
    cleaned_text = clean_html(raw_html)
    
    # 进一步清理：移除多余空白
    cleaned_text = re.sub(r'\s+', ' ', cleaned_text)
    cleaned_text = cleaned_text.strip()
    
    return cleaned_text


def preprocess_news_dict(news_dict: Dict) -> Tuple[str, str]:
    """
    预处理新闻字典，清理 HTML 内容
    
    Args:
        news_dict: 包含 publish_time, title, content 的字典
        
    Returns:
        (cleaned_text, publish_time): 清理后的文本内容和发布时间
    """
    publish_time = news_dict.get('publish_time', '')
    title = news_dict.get('title', '')
    content = news_dict.get('content', '')
    
    # 清理标题和内容的 HTML
    title_text = preprocess_html(title) if title else ""
    content_text = preprocess_html(content) if content else ""
    
    # 组合标题和内容
    if title_text and content_text:
        cleaned_text = f"{title_text}\n\n{content_text}"
    elif title_text:
        cleaned_text = title_text
    elif content_text:
        cleaned_text = content_text
    else:
        cleaned_text = ""
    
    return cleaned_text, publish_time


def analyze(news_data: Union[str, Dict, List[Dict]]) -> Optional[Union[Dict, List[Dict]]]:
    """
    分析新闻内容，生成摘要、情绪和影响强度
    
    Args:
        news_data: 可以是以下格式之一：
            - 字符串：原始 HTML 格式的新闻内容
            - 字典：包含 publish_time, title, content 的新闻字典
            - 列表：包含多个新闻字典的列表
        
    Returns:
        如果输入是字符串或单个字典，返回包含以下字段的字典：
        - paraphrase: 新闻主题/摘要
        - sentiment: 情绪方向 [-10, 10]
        - impact: 影响强度 [0, 10]
        
        如果输入是列表，返回字典列表
        
        如果分析失败返回 None
    """
    # 处理列表输入
    if isinstance(news_data, list):
        if not news_data:
            logger.warning("输入的新闻列表为空")
            return None
        
        results = []
        total = len(news_data)
        logger.info(f"开始批量分析 {total} 条新闻...")
        
        for i, news_item in enumerate(news_data, 1):
            try:
                logger.info(f"正在分析第 {i}/{total} 条新闻...")
                result = analyze(news_item)
                if result:
                    results.append(result)
                    logger.info(f"第 {i} 条新闻分析成功")
                else:
                    logger.warning(f"第 {i} 条新闻分析失败，返回 None")
            except Exception as e:
                logger.error(f"第 {i} 条新闻分析时发生异常: {e}", exc_info=True)
                # 继续处理下一条新闻，不中断整个流程
        
        logger.info(f"批量分析完成: 成功 {len(results)}/{total} 条")
        return results if results else None
    
    # 处理字典输入
    if isinstance(news_data, dict):
        publish_time = news_data.get('publish_time', '')
        title = news_data.get('title', '')
        content = news_data.get('content', '')
        
        if not title and not content:
            logger.warning("新闻字典中 title 和 content 都为空")
            return None
        
        logger.info(f"开始预处理新闻: 标题={title[:50]}..., 发布时间={publish_time}")
        cleaned_text, publish_time = preprocess_news_dict(news_data)
        
        if not cleaned_text:
            logger.warning("清理后的文本为空")
            return None
        
        logger.info(f"清理后的文本长度: {len(cleaned_text)} 字符")
    
    # 处理字符串输入（向后兼容）
    elif isinstance(news_data, str):
        if not news_data:
            logger.warning("输入的 HTML 内容为空")
            return None
        
        logger.info("开始预处理 HTML 内容...")
        cleaned_text = preprocess_html(news_data)
        
        if not cleaned_text:
            logger.warning("清理后的文本为空")
            return None
        
        logger.info(f"清理后的文本长度: {len(cleaned_text)} 字符")
    
    else:
        logger.error(f"不支持的数据类型: {type(news_data)}")
        return None
    
    try:
        # 延迟导入 openai（只在需要时导入）
        try:
            from openai import OpenAI
        except ImportError:
            logger.error("未安装 openai 模块，无法进行新闻分析。请运行: pip install openai")
            return None
        
        # 初始化 DeepSeek API 客户端
        logger.info("正在初始化 DeepSeek API 客户端...")
        api_key = get_deepseek_api_key()
        client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")
        
        # 构建提示词
        prompt = PROMPT.format(content=cleaned_text)
        
        # 调用 API
        logger.info("正在调用 DeepSeek API 进行分析...")
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": "You are a professional financial news analyst specializing in analyzing stock market-related news."},
                {"role": "user", "content": prompt}
            ],
            stream=False,
            temperature=0.3  # Lower temperature for more consistent results
        )
        
        content = response.choices[0].message.content
        logger.info("成功收到 API 响应")
        
        # 解析 JSON 响应
        # 先移除 markdown 代码块标记
        json_str = content.strip()
        json_str = re.sub(r'```json\s*', '', json_str)
        json_str = re.sub(r'```\s*', '', json_str)
        json_str = json_str.strip()
        
        # 尝试提取 JSON 对象（查找第一个 { 到最后一个 } 之间的内容）
        # 先尝试直接解析
        result = None
        try:
            result = json.loads(json_str)
        except json.JSONDecodeError:
            # 如果直接解析失败，尝试提取 JSON 对象
            start_idx = json_str.find('{')
            end_idx = json_str.rfind('}')
            if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
                json_str = json_str[start_idx:end_idx + 1]
                try:
                    result = json.loads(json_str)
                except json.JSONDecodeError:
                    logger.error(f"无法解析 JSON，原始内容: {content[:500]}")
                    raise
        
        if result is None:
            raise json.JSONDecodeError("无法解析 JSON 响应", json_str, 0)
        
        # 验证和规范化结果
        paraphrase = result.get('paraphrase', '').strip()
        
        # 处理 sentiment，添加错误处理
        try:
            sentiment = int(result.get('sentiment', 0))
        except (ValueError, TypeError) as e:
            logger.warning(f"sentiment 值无效: {result.get('sentiment')}，使用默认值 0。错误: {e}")
            sentiment = 0
        
        # 处理 impact，添加错误处理
        try:
            impact = int(result.get('impact', 0))
        except (ValueError, TypeError) as e:
            logger.warning(f"impact 值无效: {result.get('impact')}，使用默认值 0。错误: {e}")
            impact = 0
        
        # 确保数值在有效范围内
        sentiment = max(-10, min(10, sentiment))
        impact = max(0, min(10, impact))
        
        # 验证 paraphrase 不为空
        if not paraphrase:
            logger.warning("paraphrase 为空，使用默认值")
            paraphrase = "No summary available"
        
        analysis_result = {
            'paraphrase': paraphrase,
            'sentiment': sentiment,
            'impact': impact
        }
        
        logger.info(f"分析完成: sentiment={sentiment}, impact={impact}, paraphrase长度={len(paraphrase)}")
        
        return analysis_result
        
    except json.JSONDecodeError as e:
        logger.error(f"JSON 解析失败: {e}")
        logger.error(f"响应内容: {content}")
        return None
    except ValueError as e:
        logger.error(f"配置错误: {e}")
        return None
    except Exception as e:
        logger.error(f"分析失败: {e}", exc_info=True)
        return None


def analyze_news():
    """
    从数据库读取第一条新闻并进行分析（demo 函数）
    
    Returns:
        分析结果字典，如果失败返回 None
    """
    import sqlite3
    
    try:
        logger.info("正在从数据库读取第一条新闻...")
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
            return None
        
        logger.info(f"找到新闻记录: ID={row['id']}, stock_code={row['stock_code']}")
        
        # 调试：打印原始新闻数据的前500个字符
        news_raw = row['news']
        logger.debug(f"原始新闻数据前500字符: {news_raw[:500] if news_raw else 'None'}")
        
        # 解析新闻 JSON
        news_obj = parse_news_json(news_raw)
        
        if not news_obj:
            logger.error(f"无法解析新闻 JSON。原始数据: {news_raw[:200] if news_raw else 'None'}")
            conn.close()
            return None
        
        # 构建新闻字典
        news_dict = {
            'publish_time': news_obj.get('publish_time', ''),
            'title': news_obj.get('title', ''),
            'content': news_obj.get('content', '')
        }
        
        if not news_dict['title'] and not news_dict['content']:
            logger.error("新闻内容为空")
            conn.close()
            return None
        
        # 进行分析
        logger.info("开始分析新闻...")
        result = analyze(news_dict)
        
        conn.close()
        
        if result:
            print("\n" + "=" * 80)
            print("新闻分析结果")
            print("=" * 80)
            print(f"股票代码: {row['stock_code']}")
            print(f"记录 ID: {row['id']}")
            print(f"发布时间: {news_dict['publish_time']}")
            print(f"标题: {news_dict['title']}")
            print("\n分析结果:")
            print(f"  主题/摘要: {result['paraphrase']}")
            print(f"  情绪方向: {result['sentiment']} (范围: -10 到 10)")
            print(f"  影响强度: {result['impact']} (范围: 0 到 10)")
            print("=" * 80)
            
            # 输出 JSON 格式
            print("\nJSON 格式:")
            print(json.dumps(result, indent=2, ensure_ascii=False))
        
        return result
        
    except Exception as e:
        logger.error(f"分析新闻失败: {e}", exc_info=True)
        return None


if __name__ == "__main__":
    # Demo: 分析数据库中的第一条新闻
    analyze_news()
