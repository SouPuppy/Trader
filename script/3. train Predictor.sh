#!/bin/bash
# LSTM 模型训练脚本
# 训练共享参数的 LSTM 模型（前21天预测后1天）
# 用10支股票训练，10支股票测试

# 获取脚本所在目录的父目录（项目根目录）
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( cd "$SCRIPT_DIR/.." && pwd )"

# 切换到项目根目录
cd "$PROJECT_ROOT" || exit 1

# 设置 PYTHONPATH 为项目根目录，这样 Python 可以找到 trader 模块
export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"

# 检查是否安装了 wandb（在 poetry 环境中）
USE_WANDB=""
if poetry run python -c "import wandb" 2>/dev/null; then
    echo "检测到 wandb 已安装，默认启用 wandb 记录"
    echo "提示: 如果未登录 wandb，请先运行: poetry run wandb login"
    echo "     如果想禁用 wandb，请修改脚本中的 USE_WANDB 变量"
    echo ""
    # 默认启用 wandb
    USE_WANDB="--wandb"
else
    echo "警告: wandb 未安装，将不使用 wandb 记录"
    echo "安装命令: poetry add wandb"
    echo ""
fi

# 运行训练脚本
echo "开始训练共享 LSTM 模型..."
echo "模型: 用前21天的 close_price 预测后1天的 close_price"
echo "训练集: 10支股票（见 model_config.toml）"
echo "测试集: 10支股票（见 model_config.toml）"
echo ""

poetry run python -m trader.predictor.train \
    $USE_WANDB

echo ""
echo "训练完成！"
echo "模型保存位置: weights/LSTM/"
echo "  - model.pth (模型权重)"
echo "  - scaler.pkl (标准化器)"

