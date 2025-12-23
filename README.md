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
