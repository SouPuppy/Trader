#!/bin/bash
# 可视化 Predictor 预测结果
# 显示所有已训练股票的预测结果

cd "$(dirname "$0")/.." || exit

echo "=========================================="
echo "可视化 Predictor 预测结果"
echo "=========================================="
echo ""
echo "用法: ./script/C. visualize Predictor.sh [输出目录]"
echo "示例: ./script/C. visualize Predictor.sh ./output/predictor"
echo ""

OUTPUT_DIR="${1:-./output/predictor}"

echo "开始生成预测结果可视化图表..."
poetry run python -m trader.predictor.visualize --output "$OUTPUT_DIR"

echo ""
echo "=========================================="
echo "图表已保存到: $OUTPUT_DIR"
echo "=========================================="
