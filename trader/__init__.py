"""
Trader 模块
提供全局配置导入
"""

from trader.config import PROJECT_ROOT, DB_PATH, CSV_PATH

__all__ = ['PROJECT_ROOT', 'DB_PATH', 'CSV_PATH']
