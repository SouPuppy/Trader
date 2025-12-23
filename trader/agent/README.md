# TradingAgent 模块文档

## 核心概念

TradingAgent 模块区分了两个关键概念：

### Score（研究问题）
- **定义**: 表达"看好程度/预期收益/排序依据"
- **范围**: 通常为 `[-1, 1]` 或 `[0, 1]`
- **用途**: 
  - 排序和筛选股票
  - 表达对股票的看好/看空程度
  - 正数表示看好，负数表示看空
  - 绝对值越大表示看好/看空程度越高
- **特点**: 纯粹的研究问题，不考虑实际交易限制

### Weight（工程 + 风控问题）
- **定义**: 表达"实际配置多少资金"
- **范围**: `[0, 1]`，表示资金配置比例
- **用途**:
  - 决定实际买入/卖出的资金量
  - 考虑风险控制和仓位限制
  - 0 表示不配置，1 表示全仓（通常不建议）
- **特点**: 工程和风控问题，需要考虑：
  - 单个股票仓位限制
  - 总仓位限制
  - 账户风险
  - 流动性限制等

## 类层次结构

```
TradingAgent (基类)
  ├── score() - 抽象方法：计算看好程度（必须实现）
  ├── weight() - 默认实现：计算配置比例（可以重写）
  ├── get_scores() - 批量计算 score
  ├── get_weights() - 批量计算 weight
  ├── normalize_weights() - 权重归一化工具方法
  └── filter_by_score() - 根据 score 筛选股票工具方法

DummyAgent (示例实现)
  └── 继承自 TradingAgent
      └── 提供简单的 score() 实现示例
```

注意：`AbstractAgent` 是 `TradingAgent` 的别名，用于向后兼容。

## 使用示例

### 1. 创建自定义 TradingAgent

```python
from trader.agent import TradingAgent
from trader.backtest.engine import BacktestEngine

class MyAgent(TradingAgent):
    def score(self, stock_code: str, engine: BacktestEngine) -> float:
        """计算看好程度（必须实现）"""
        # 使用特征计算 score
        ret_1d = engine.get_feature("ret_1d", stock_code)
        ret_20d = engine.get_feature("ret_20d", stock_code)
        
        # 自定义计算逻辑
        score = (ret_1d * 0.3 + ret_20d * 0.7) if ret_1d and ret_20d else 0.0
        return max(-1.0, min(1.0, score))
    
    def weight(self, stock_code: str, score: float, engine: BacktestEngine) -> float:
        """计算配置比例（可选重写，默认实现基于 score 和风险参数）"""
        base_weight = super().weight(stock_code, score, engine)
        
        # 添加自定义风控逻辑
        # 例如：检查持仓数量、账户风险等
        
        return base_weight
```

### 2. 在回测中使用 TradingAgent

```python
from trader.agent import DummyAgent
from trader.backtest.engine import BacktestEngine

def on_trading_day(engine: BacktestEngine, date: str):
    # 创建 TradingAgent
    agent = DummyAgent(
        max_position_weight=0.1,  # 单个股票最多10%
        min_score_threshold=0.0,  # score >= 0 才配置
        max_total_weight=0.8      # 总配置不超过80%
    )
    
    # 候选股票列表
    stock_codes = ["AAPL.O", "MSFT.O", "GOOGL.O"]
    
    # 1. 计算所有股票的 score（研究问题）
    scores = agent.get_scores(stock_codes, engine)
    print(f"Scores: {scores}")
    # 输出: {'AAPL.O': 0.5, 'MSFT.O': 0.3, 'GOOGL.O': -0.2}
    
    # 2. 根据 score 筛选（可选）
    filtered_scores = agent.filter_by_score(scores, top_n=5)
    
    # 3. 计算 weight（工程+风控问题）
    weights = agent.get_weights(filtered_scores, engine)
    print(f"Weights: {weights}")
    # 输出: {'AAPL.O': 0.05, 'MSFT.O': 0.03, 'GOOGL.O': 0.0}
    
    # 4. 归一化权重（确保总权重不超过限制）
    normalized_weights = agent.normalize_weights(weights)
    
    # 5. 根据 weight 执行交易
    account_equity = engine.account.equity(engine.get_market_prices(stock_codes))
    for stock_code, weight in normalized_weights.items():
        if weight > 0:
            amount = account_equity * weight
            engine.buy(stock_code, amount=amount)
```

## 设计原则

1. **关注点分离**:
   - Score: 专注于研究问题（如何判断股票好坏）
   - Weight: 专注于工程和风控问题（如何配置资金）

2. **可扩展性**:
   - 继承 `TradingAgent` 可以快速实现新 TradingAgent
   - 只实现 `score()` 方法即可，`weight()` 有默认实现
   - 可以重写 `weight()` 方法添加自定义风控逻辑

3. **灵活性**:
   - 可以重写 `weight()` 方法添加自定义风控逻辑
   - 可以重写 `on_date()` 方法实现定期任务

## 注意事项

1. **Score 和 Weight 的区别**:
   - Score 高不代表 Weight 高
   - Weight 需要考虑风险控制，即使 Score 很高也可能配置较小比例

2. **日期保护**:
   - TradingAgent 通过 engine 访问数据，自动带有日期保护
   - 不能访问未来数据

3. **错误处理**:
   - `get_scores()` 和 `get_weights()` 会自动处理异常
   - 计算失败的股票会返回 0.0

