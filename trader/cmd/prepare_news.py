"""
新闻准备脚本
直接调用 trader.news.prepare 模块的功能
"""
import sys
import runpy
from pathlib import Path

# 添加项目根目录到 Python 路径
project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

if __name__ == "__main__":
    # 直接执行 prepare 模块的 __main__ 逻辑
    runpy.run_module('trader.news.prepare', run_name='__main__')

