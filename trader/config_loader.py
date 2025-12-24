"""
配置文件加载器
用于读取 trader/config.toml 配置文件
"""
import sys
from pathlib import Path
from typing import List, Optional

# 添加项目根目录到 Python 路径
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

try:
    import tomllib  # Python 3.11+
except ImportError:
    try:
        import tomli as tomllib  # 需要安装 tomli
    except ImportError:
        raise ImportError(
            "需要 Python 3.11+ 或安装 tomli 库来读取 TOML 文件。"
            "安装命令: pip install tomli"
        )

from trader.config import PROJECT_ROOT
from trader.logger import get_logger

logger = get_logger(__name__)


def load_config(config_path: Optional[Path] = None) -> dict:
    """
    加载配置文件
    
    Args:
        config_path: 配置文件路径，如果为 None 则使用默认路径 trader/config.toml
    
    Returns:
        配置字典
    """
    if config_path is None:
        config_path = PROJECT_ROOT / 'trader' / 'config.toml'
    
    if not config_path.exists():
        raise FileNotFoundError(f"配置文件不存在: {config_path}")
    
    with open(config_path, 'rb') as f:
        config = tomllib.load(f)
    
    logger.debug(f"已加载配置文件: {config_path}")
    return config


def get_test_stocks(config_path: Optional[Path] = None) -> List[str]:
    """
    获取测试股票列表
    
    Args:
        config_path: 配置文件路径
    
    Returns:
        测试股票代码列表
    """
    config = load_config(config_path)
    return config.get('test', {}).get('stocks', [])

