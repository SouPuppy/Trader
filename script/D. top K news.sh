#!/bin/bash
# 找出20支股票波动最大的10天，并标注新闻
# 从裸数据中找出波动最大的交易日，并显示对应的新闻

# 获取脚本所在目录的父目录（项目根目录）
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( cd "$SCRIPT_DIR/.." && pwd )"

# 切换到项目根目录
cd "$PROJECT_ROOT" || exit 1

# 设置 PYTHONPATH 为项目根目录，这样 Python 可以找到 trader 模块
export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"

echo "=========================================="
echo "找出股票波动最大的天数并标注新闻"
echo "=========================================="
echo ""
echo "默认: 分析20支股票，每支股票找出波动最大的10天"
echo "用法: $0 [--num-stocks N] [--top-days M]"
echo "示例: $0 --num-stocks 15 --top-days 20"
echo ""

# 运行分析脚本，传递所有参数
poetry run python -m trader.debug.find_volatile_days "$@"

echo ""
echo "=========================================="
echo "分析完成！"
echo "结果保存在: output/debug/volatile_days_*stocks_*days.json"
echo "=========================================="

