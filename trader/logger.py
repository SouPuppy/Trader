"""
专业的日志系统配置模块
提供统一的日志接口，支持文件和控制台输出
支持彩色日志输出
"""
import logging
import os
import sys
from pathlib import Path
from logging.handlers import RotatingFileHandler

# ANSI 颜色代码
class Colors:
    """ANSI 颜色代码"""
    RESET = '\033[0m'
    BOLD = '\033[1m'
    
    # 文本颜色
    BLACK = '\033[30m'
    RED = '\033[31m'
    GREEN = '\033[32m'
    YELLOW = '\033[33m'
    BLUE = '\033[34m'
    MAGENTA = '\033[35m'
    CYAN = '\033[36m'
    WHITE = '\033[37m'
    
    # 亮色
    BRIGHT_BLACK = '\033[90m'
    BRIGHT_RED = '\033[91m'
    BRIGHT_GREEN = '\033[92m'
    BRIGHT_YELLOW = '\033[93m'
    BRIGHT_BLUE = '\033[94m'
    BRIGHT_MAGENTA = '\033[95m'
    BRIGHT_CYAN = '\033[96m'
    BRIGHT_WHITE = '\033[97m'

# 日志级别颜色映射
LEVEL_COLORS = {
    'DEBUG': Colors.BRIGHT_BLACK,
    'INFO': Colors.BRIGHT_GREEN,
    'WARNING': Colors.BRIGHT_YELLOW,
    'ERROR': Colors.BRIGHT_RED,
    'CRITICAL': Colors.BRIGHT_RED + Colors.BOLD,
}

class ColoredFormatter(logging.Formatter):
    """带颜色的日志格式化器，支持级别名称和模块名称对齐"""
    
    # 级别名称固定宽度（对齐用）
    LEVEL_WIDTH = 8
    # 模块名称固定宽度（对齐用）
    NAME_WIDTH = 25
    
    def __init__(self, use_color=True, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # 检查是否支持颜色：
        # 1. 环境变量 NO_COLOR 存在则禁用颜色
        # 2. 环境变量 FORCE_COLOR 存在则强制启用颜色
        # 3. 默认启用颜色（大多数现代终端都支持）
        no_color = os.environ.get('NO_COLOR', '').lower() in ('1', 'true', 'yes')
        force_color = os.environ.get('FORCE_COLOR', '').lower() in ('1', 'true', 'yes')
        
        if no_color:
            self.use_color = False
        elif force_color or use_color:
            # 默认启用颜色，除非明确禁用
            self.use_color = True
        else:
            self.use_color = False
    
    def format(self, record):
        # 保存原始值
        original_levelname = record.levelname
        original_name = record.name
        
        # 对齐级别名称（固定宽度）
        aligned_levelname = original_levelname.ljust(self.LEVEL_WIDTH)
        
        # 对齐模块名称（固定宽度）
        aligned_name = original_name.ljust(self.NAME_WIDTH)
        
        # 如果使用颜色，为级别名称添加颜色
        if self.use_color:
            color = LEVEL_COLORS.get(original_levelname, Colors.RESET)
            # 先对齐，再添加颜色
            record.levelname = f"{color}{aligned_levelname}{Colors.RESET}"
        else:
            # 不使用颜色时，只对齐
            record.levelname = aligned_levelname
        
        # 设置对齐后的模块名称
        record.name = aligned_name
        
        # 格式化日志消息
        formatted = super().format(record)
        
        # 恢复原始值（避免影响其他 handler）
        record.levelname = original_levelname
        record.name = original_name
        
        return formatted

# 获取项目根目录（独立定义，避免循环导入）
PROJECT_ROOT = Path(__file__).parent.parent
LOG_DIR = PROJECT_ROOT / 'logs'
LOG_DIR.mkdir(exist_ok=True)

# 日志格式
LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
DATE_FORMAT = '%Y-%m-%d %H:%M:%S'

# 日志文件路径（使用固定文件名，每次运行追加）
LOG_FILE = LOG_DIR / 'trader.log'

def setup_logger(name: str = 'trader', level: str = 'INFO') -> logging.Logger:
    """
    设置并返回配置好的 logger
    
    Args:
        name: logger 名称，默认为 'trader'
        level: 日志级别，可选 'DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'
        
    Returns:
        配置好的 logger 实例
    """
    logger = logging.getLogger(name)
    
    # 避免重复添加 handler
    if logger.handlers:
        return logger
    
    # 设置日志级别
    log_level = getattr(logging, level.upper(), logging.INFO)
    logger.setLevel(log_level)
    
    # 控制台 handler（带颜色）
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level)
    console_formatter = ColoredFormatter(
        use_color=True,
        fmt=LOG_FORMAT,
        datefmt=DATE_FORMAT
    )
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)
    
    # 文件 handler（带轮转，不使用颜色，但使用对齐）
    file_handler = RotatingFileHandler(
        LOG_FILE,
        maxBytes=10 * 1024 * 1024,  # 10MB
        backupCount=5,
        encoding='utf-8'
    )
    file_handler.setLevel(log_level)
    file_formatter = ColoredFormatter(use_color=False, fmt=LOG_FORMAT, datefmt=DATE_FORMAT)  # 文件不使用颜色，但使用对齐
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)
    
    # 防止日志传播到根 logger
    logger.propagate = False
    
    return logger

def get_logger(name: str = None) -> logging.Logger:
    """
    获取 logger 实例
    
    Args:
        name: logger 名称，如果为 None 则使用调用模块的名称
        
    Returns:
        logger 实例
    """
    if name is None:
        # 自动获取调用模块的名称
        import inspect
        frame = inspect.currentframe().f_back
        module_name = frame.f_globals.get('__name__', 'trader')
        name = module_name.split('.')[0] if '.' in module_name else module_name
    
    return setup_logger(name)

# 创建默认 logger
logger = setup_logger()


def log_separator(width: int = 60, char: str = "=", indent: str = "\t"):
    """
    生成分隔线字符串（带缩进）
    
    Args:
        width: 分隔线宽度
        char: 分隔线字符
        indent: 缩进字符串（默认使用 tab）
        
    Returns:
        str: 分隔线字符串
    """
    return indent + char * width


def log_section(title: str, width: int = 60, char: str = "=", indent: str = "\t"):
    """
    生成带标题的分隔线（带缩进）
    
    Args:
        title: 标题文本
        width: 分隔线宽度
        char: 分隔线字符
        indent: 缩进字符串（默认使用 tab）
        
    Returns:
        List[str]: [分隔线, 标题, 分隔线] 列表
    """
    sep = indent + char * width
    title_line = indent + title
    return [sep, title_line, sep]

