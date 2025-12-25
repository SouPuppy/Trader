#!/bin/bash

# 退出 wandb 登录的脚本

echo "正在退出 wandb 登录..."

# 方法 1: 使用 wandb logout 命令（如果可用）
if command -v wandb &> /dev/null; then
    echo "方法 1: 使用 wandb logout 命令"
    wandb logout
    echo "✓ 已执行 wandb logout"
fi

# 方法 2: 删除 wandb 配置文件
WANDB_CONFIG_DIR="$HOME/.config/wandb"
if [ -d "$WANDB_CONFIG_DIR" ]; then
    echo ""
    echo "方法 2: 删除 wandb 配置文件"
    echo "找到 wandb 配置目录: $WANDB_CONFIG_DIR"
    
    # 备份设置文件（可选）
    if [ -f "$WANDB_CONFIG_DIR/settings" ]; then
        echo "备份设置文件..."
        cp "$WANDB_CONFIG_DIR/settings" "$WANDB_CONFIG_DIR/settings.backup"
    fi
    
    # 删除设置文件中的 base_url 和 api_key
    if [ -f "$WANDB_CONFIG_DIR/settings" ]; then
        echo "清理设置文件..."
        # 删除包含 base_url 和 api_key 的行
        sed -i.backup '/^base_url/d' "$WANDB_CONFIG_DIR/settings" 2>/dev/null || \
        sed -i '' '/^base_url/d' "$WANDB_CONFIG_DIR/settings" 2>/dev/null || true
        sed -i.backup '/^api_key/d' "$WANDB_CONFIG_DIR/settings" 2>/dev/null || \
        sed -i '' '/^api_key/d' "$WANDB_CONFIG_DIR/settings" 2>/dev/null || true
        echo "✓ 已清理设置文件"
    fi
fi

# 方法 3: 删除 .netrc 中的 wandb 相关条目
NETRC_FILE="$HOME/.netrc"
if [ -f "$NETRC_FILE" ]; then
    echo ""
    echo "方法 3: 检查 .netrc 文件"
    echo "找到 .netrc 文件: $NETRC_FILE"
    
    # 检查是否包含 wandb.ai
    if grep -q "wandb.ai" "$NETRC_FILE" 2>/dev/null; then
        echo "发现 wandb.ai 条目，正在备份并删除..."
        cp "$NETRC_FILE" "$NETRC_FILE.backup"
        
        # 删除包含 wandb.ai 的机器条目
        # 使用 awk 来删除从 "machine wandb.ai" 到下一个 "machine" 或文件结尾之间的所有行
        awk '
        /^machine wandb.ai/ { skip=1; next }
        /^machine / { skip=0 }
        !skip { print }
        ' "$NETRC_FILE.backup" > "$NETRC_FILE"
        
        echo "✓ 已从 .netrc 删除 wandb.ai 条目"
        echo "  备份文件: $NETRC_FILE.backup"
    else
        echo "  .netrc 文件中未找到 wandb.ai 条目"
    fi
fi

echo ""
echo "=========================================="
echo "退出 wandb 登录完成！"
echo ""
echo "验证方法："
echo "  1. 运行训练脚本，应该会使用离线模式"
echo "  2. 检查日志中应该显示 '离线模式' 而不是 '在线模式'"
echo ""
echo "如果需要重新登录，运行:"
echo "  wandb login"
echo "  或"
echo "  poetry run wandb login"
echo "=========================================="


