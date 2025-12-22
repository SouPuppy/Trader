import os
import sys
from pathlib import Path

# 添加项目根目录到 Python 路径
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from trader.logger import get_logger

logger = get_logger(__name__)

if __name__ == "__main__":
    # test env
    api_key = os.getenv("DEEPSEEK_API_KEY")
    if api_key:
        logger.info("DEEPSEEK_API_KEY 已设置")
        logger.debug(f"DEEPSEEK_API_KEY: {api_key[:10]}...")  # 只显示前10个字符
    else:
        logger.warning("DEEPSEEK_API_KEY 未设置")