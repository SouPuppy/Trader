#!/bin/bash
# 生成特征可视化图表

cd "$(dirname "$0")/.." || exit

echo "=========================================="
echo "生成特征可视化图表"
echo "=========================================="
echo ""
echo "用法: ./script/visualize.sh [输出目录]"
echo "示例: ./script/visualize.sh ./output/features"
echo ""

OUTPUT_DIR="${1:-./output/features}"

python -m trader.cmd.build_features --list

echo ""
echo "开始生成图表..."
python -m trader.visualize.daily_features --output "$OUTPUT_DIR"

echo ""
echo "=========================================="
echo "图表已保存到: $OUTPUT_DIR"
echo "=========================================="
