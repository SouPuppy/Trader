#!/bin/bash
# This script initializes the `data.sqlite3` database from `data.csv`

# 获取脚本所在目录的父目录（项目根目录）
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( cd "$SCRIPT_DIR/.." && pwd )"

# 切换到项目根目录
cd "$PROJECT_ROOT" || exit 1

# 设置 PYTHONPATH 为项目根目录，这样 Python 可以找到 trader 模块
export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"

# 检查 CSV 文件是否存在
if [ ! -f "data/data.csv" ]; then
    echo "错误: CSV 文件不存在: data/data.csv"
    exit 1
fi

# 运行数据库初始化脚本
poetry run python trader/cmd/db_init.py

# 检查执行结果
if [ $? -eq 0 ]; then
    echo ""
    echo "😎 数据库初始化成功！"
    echo "数据库位置: $PROJECT_ROOT/data/data.sqlite3"
else
    echo ""
    echo "🤯 数据库初始化失败！"
    exit 1
fi
