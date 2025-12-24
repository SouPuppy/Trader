"""
配置文件加载器
用于读取 model_config.toml 配置文件
"""
import sys
from pathlib import Path
from typing import Dict, List, Optional

# 添加项目根目录到 Python 路径
project_root = Path(__file__).parent.parent.parent
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


def load_model_config(config_path: Optional[Path] = None) -> Dict:
    """
    加载模型配置文件
    
    Args:
        config_path: 配置文件路径，如果为 None 则使用默认路径
    
    Returns:
        配置字典，包含以下键：
        - stocks: 所有股票列表
        - train: 训练集股票列表
        - test: 测试集股票列表
        - model: 模型超参数
        - training: 训练参数
        - data: 数据配置
    """
    if config_path is None:
        config_path = PROJECT_ROOT / 'trader' / 'predictor' / 'model_config.toml'
    
    if not config_path.exists():
        raise FileNotFoundError(f"配置文件不存在: {config_path}")
    
    with open(config_path, 'rb') as f:
        config = tomllib.load(f)
    
    logger.info(f"已加载配置文件: {config_path}")
    return config


def get_train_stocks(config_path: Optional[Path] = None) -> List[str]:
    """
    获取训练集股票列表
    
    Args:
        config_path: 配置文件路径
    
    Returns:
        训练集股票代码列表
    """
    config = load_model_config(config_path)
    return config.get('train', {}).get('stocks', [])


def get_test_stocks(config_path: Optional[Path] = None) -> List[str]:
    """
    获取测试集股票列表
    
    Args:
        config_path: 配置文件路径
    
    Returns:
        测试集股票代码列表
    """
    config = load_model_config(config_path)
    return config.get('test', {}).get('stocks', [])


def get_all_stocks(config_path: Optional[Path] = None) -> List[str]:
    """
    获取所有股票列表
    
    Args:
        config_path: 配置文件路径
    
    Returns:
        所有股票代码列表
    """
    config = load_model_config(config_path)
    return config.get('stocks', {}).get('all', [])


def get_model_config(config_path: Optional[Path] = None) -> Dict:
    """
    获取模型超参数配置
    
    Args:
        config_path: 配置文件路径
    
    Returns:
        模型配置字典
    """
    config = load_model_config(config_path)
    return config.get('model', {})


def get_training_config(config_path: Optional[Path] = None) -> Dict:
    """
    获取训练参数配置
    
    Args:
        config_path: 配置文件路径
    
    Returns:
        训练配置字典
    """
    config = load_model_config(config_path)
    return config.get('training', {})


def get_data_config(config_path: Optional[Path] = None) -> Dict:
    """
    获取数据配置
    
    Args:
        config_path: 配置文件路径
    
    Returns:
        数据配置字典
    """
    config = load_model_config(config_path)
    return config.get('data', {})


if __name__ == '__main__':
    # 测试配置加载
    try:
        config = load_model_config()
        print("配置加载成功！")
        print(f"\n所有股票 ({len(get_all_stocks())} 支):")
        for stock in get_all_stocks():
            print(f"  - {stock}")
        
        print(f"\n训练集股票 ({len(get_train_stocks())} 支):")
        for stock in get_train_stocks():
            print(f"  - {stock}")
        
        print(f"\n测试集股票 ({len(get_test_stocks())} 支):")
        for stock in get_test_stocks():
            print(f"  - {stock}")
        
        print(f"\n模型配置:")
        model_config = get_model_config()
        for key, value in model_config.items():
            print(f"  {key}: {value}")
        
        print(f"\n训练配置:")
        training_config = get_training_config()
        for key, value in training_config.items():
            print(f"  {key}: {value}")
            
    except Exception as e:
        logger.error(f"加载配置失败: {e}", exc_info=True)

