### Requirements

- Poetry

### 项目结构

- `cmd/` 可执行文件

### Quick Start

1. 配置环境变量

    ```bash
    # 复制环境变量模板
    cp .env.example .env
    
    # 编辑 .env 文件, 填入你的 DEEPSEEK API Key
    # DEEPSEEK_API_KEY=your_api_key_here
    ```

2. 初始化数据库

    ```bash
    ./script/init_db.sh
    ```

3. 准备数据

    ```bash
    ./script/prepare_news.sh
    ```

4. 运行脚本

    ```bash
    ./script/run.sh
    ```

## 运行实验

项目提供了多个交易策略实验，位于 `experiments/` 目录下。

### 交互式运行实验

使用实验运行器，交互式选择并运行实验：

```bash
./script/B. run experiment.sh
```

这会显示所有可用的实验列表, 你可以选择要运行的实验

### 实验输出

实验运行后会在 `output/backtest/` 目录下生成回测报告，包括:
- 交易记录
- 账户权益曲线
- 性能指标
- 可视化图表
