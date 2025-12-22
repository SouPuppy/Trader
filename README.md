### Requirements

- Poetry

### 项目结构

- `cmd/` 可执行文件

### Quick Start

1. 配置环境变量

    ```bash
    # 复制环境变量模板
    cp .env.example .env
    
    # 编辑 .env 文件，填入你的 DEEPSEEK API Key
    # DEEPSEEK_API_KEY=your_api_key_here
    ```

2. 初始化数据库

    ```bash
    ./script/init_db.sh
    ```

3. 执行策略

    ```bash
    ./script/run.sh
    ```

4. [其他] 运行脚本

    ```bash
    poetry run python <module>
    ```

### 环境变量配置

项目会自动从 `.env` 文件加载环境变量。主要配置项：

- `DEEPSEEK_API_KEY`: DEEPSEEK API 密钥（必需）

在代码中使用：

```python
from trader.config import get_deepseek_api_key

api_key = get_deepseek_api_key()
```
