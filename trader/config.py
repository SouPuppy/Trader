"""
配置模块：自动从 .env 文件加载环境变量
提供全局路径配置
"""
import os
from pathlib import Path
from dotenv import load_dotenv
from trader.logger import get_logger

logger = get_logger(__name__)

# 获取项目根目录
PROJECT_ROOT = Path(__file__).parent.parent

# 数据库路径配置
DB_PATH = PROJECT_ROOT / 'data' / 'data.sqlite3'
CSV_PATH = PROJECT_ROOT / 'data' / 'data.csv'

# 确保必要的目录存在
PROJECT_ROOT.mkdir(exist_ok=True)
(PROJECT_ROOT / 'data').mkdir(exist_ok=True)

# 自动加载 .env 文件
env_path = PROJECT_ROOT / '.env'
if env_path.exists():
    try:
        load_dotenv(env_path)
        logger.debug(f"已加载环境变量文件: {env_path}")
    except Exception as e:
        # 在某些受限环境（沙盒/CI）中，.env 可能存在但不可读；此时不要让程序直接崩溃
        logger.warning(f"无法加载 .env 文件（将忽略并继续运行）: {env_path}, error={e}")
else:
    # .env 不是强依赖，缺失时保持安静（需要密钥时再报错）
    logger.debug(f".env 文件不存在: {env_path}")

def get_deepseek_api_key() -> str:
    api_key = os.getenv('DEEPSEEK_API_KEY')
    if not api_key:
        logger.error("DEEPSEEK_API_KEY is not set in `.env` file or environment variables")
        raise ValueError(
            "DEEPSEEK_API_KEY is not set in `.env` file or environment variables"
        )
    logger.debug("成功获取 DEEPSEEK_API_KEY")
    return api_key

def get_env(key: str, default: str = None) -> str:
    """
    获取环境变量
    
    Args:
        key: 环境变量名
        default: 默认值（如果不存在）
        
    Returns:
        str: 环境变量值或默认值
    """
    return os.getenv(key, default)

