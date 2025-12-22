#!/bin/bash

# 获取脚本所在目录的父目录（项目根目录）
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( cd "$SCRIPT_DIR/.." && pwd )"

# 切换到项目根目录
cd "$PROJECT_ROOT" || exit 1

# 设置 PYTHONPATH 为项目根目录，这样 Python 可以找到 trader 模块
export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"

# 运行主程序
poetry run python trader/main.py "$@"