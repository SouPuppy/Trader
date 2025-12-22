import sys
from pathlib import Path

# 添加项目根目录到 Python 路径
project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from trader.config import get_deepseek_api_key
from trader.logger import get_logger
from openai import OpenAI

logger = get_logger(__name__)

try:
    # 自动从 .env 文件加载 API Key
    logger.info("正在获取 DEEPSEEK API Key...")
    api_key = get_deepseek_api_key()
    
    logger.info("正在初始化 OpenAI 客户端...")
    client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")
    
    logger.info("正在发送请求到 DEEPSEEK API...")
    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {"role": "system", "content": "You are a helpful assistant"},
            {"role": "user", "content": "Hello"},
        ],
        stream=False
    )
    
    content = response.choices[0].message.content
    logger.info("成功收到 DEEPSEEK API 响应")
    logger.info(f"响应内容: {content}")
    
except ValueError as e:
    logger.error(f"配置错误: {e}")
    raise
except Exception as e:
    logger.error(f"API 调用失败: {e}", exc_info=True)
    raise