# Risk 风险管理模块

风险管理模块提供统一的订单意图结构和风险管理接口，用于在交易执行前进行风险控制。

## 模块结构

```
trader/risk/
├── __init__.py                    # 模块导出
├── OrderIntent.py                 # 订单意图结构定义
├── RiskManager.py                 # 风险管理器抽象接口和管道
└── control_leverage_limit.py      # 杠杆限制风险管理器实现
```

## 工作流程

```
Agent (score/weight) 
  → OrderIntent(s) 
  → RiskManager Pipeline (validate/adjust/pre_trade) 
  → Approved/Adjusted Orders 
  → Engine Execute
```

## 核心组件

### OrderIntent.py

定义统一的订单表示结构。

**OrderSide**: 订单方向枚举
- `BUY`: 买入
- `SELL`: 卖出

**PriceType**: 价格类型枚举
- `MKT`: 市价单
- `LMT`: 限价单

**OrderIntent**: 订单意图数据类

必需字段：
- `symbol`: 股票代码
- `side`: 订单方向
- `timestamp`: 交易日（YYYY-MM-DD）

数量字段（二选一）：
- `qty`: 股数（用于 SELL 或精确买入）
- `target_weight`: 目标权重 0-1（用于 BUY）

可选字段：
- `price_type`: 价格类型（默认 MKT）
- `limit_price`: 限价单价格
- `agent_name`: 生成来源
- `prompt_version`: Prompt 版本
- `confidence`: 置信度 0-1
- `reason_tags`: 理由标签列表
- `metadata`: 元数据字典

### RiskManager.py

提供统一的风险管理接口。

**RiskContext**: 风险控制上下文
- `account`: 账户信息
- `market`: 市场信息
- `current_date`: 当前交易日
- `market_prices`: 当前市场价格字典
- `metadata`: 额外上下文信息

辅助方法：
- `get_account_equity()`: 获取账户总权益
- `get_position_value(symbol)`: 获取持仓市值
- `get_position_weight(symbol)`: 获取持仓权重

**RiskManager**: 抽象基类，提供四种能力

1. `validate(ctx, order) -> (ok, reason)`
   - 验证订单是否可以通过
   - 返回 (True, None) 或 (False, "reason")

2. `adjust(ctx, order) -> order'`
   - 调整订单（削减仓位、截断换手等）
   - 返回调整后的订单意图

3. `pre_trade(ctx, orders) -> orders'`
   - 组合层面统一缩放
   - 返回调整后的订单列表

4. `post_trade(ctx, fills)`
   - 记录风控状态（冷却期、连续亏损计数等）

**RiskManagerPipeline**: 风险管理器管道
- 按顺序应用多个风险管理器
- 支持 validate、adjust、pre_trade、post_trade 的管道化处理

### control_leverage_limit.py

杠杆限制风险管理器实现。

限制账户总杠杆率（总持仓市值 / 账户权益），防止过度杠杆。

参数：
- `max_leverage`: 最大杠杆率
  - 1.0: 不允许杠杆
  - 2.0: 允许 2 倍杠杆

工作原理：
- `validate()`: 计算当前杠杆率，估算订单执行后的杠杆率，超过上限则拒绝
- `adjust()`: 如果订单会导致杠杆超限，削减订单规模

## 使用示例

### 基本使用

```python
from trader.risk import (
    OrderIntent, OrderSide, PriceType,
    RiskManagerPipeline, RiskContext,
    LeverageLimitRiskManager
)

# 创建风险管理器管道
pipeline = RiskManagerPipeline([
    LeverageLimitRiskManager(max_leverage=1.0)
])

# 创建订单意图
order = OrderIntent(
    symbol="AAPL.O",
    side=OrderSide.BUY,
    timestamp="2023-01-03",
    target_weight=0.1,
    price_type=PriceType.MKT,
    agent_name="MyAgent"
)

# 创建风险控制上下文
ctx = RiskContext(
    account=account,
    market=market,
    current_date="2023-01-03",
    market_prices={"AAPL.O": 150.0}
)

# 验证订单
ok, reason = pipeline.validate(ctx, order)

# 调整订单
adjusted_order = pipeline.adjust(ctx, order)

# 组合层面调整
adjusted_orders = pipeline.pre_trade(ctx, [adjusted_order])
```

### 在 Agent 中使用

```python
from trader.risk import RiskManagerPipeline, RiskContext, LeverageLimitRiskManager

class MyAgent:
    def __init__(self):
        self.risk_pipeline = RiskManagerPipeline([
            LeverageLimitRiskManager(max_leverage=1.0)
        ])
    
    def on_date(self, engine, date):
        # 生成订单意图
        order_intents = self.generate_order_intents(engine)
        
        # 创建风险控制上下文
        ctx = RiskContext(
            account=engine.account,
            market=engine.market,
            current_date=engine.current_date,
            market_prices=engine.get_market_prices(["AAPL.O"])
        )
        
        # 验证和调整订单
        validated_orders = [
            order for order in order_intents
            if self.risk_pipeline.validate(ctx, order)[0]
        ]
        
        adjusted_orders = [
            self.risk_pipeline.adjust(ctx, order)
            for order in validated_orders
        ]
        
        final_orders = self.risk_pipeline.pre_trade(ctx, adjusted_orders)
        
        # 执行订单
        self.execute_orders(final_orders, engine)
```

### 创建自定义风险管理器

```python
from trader.risk import RiskManager, RiskContext, OrderIntent
from trader.risk.OrderIntent import OrderSide
from typing import Tuple, Optional

class PositionLimitRiskManager(RiskManager):
    """持仓数量限制风险管理器"""
    
    def __init__(self, name: str = "PositionLimit", max_positions: int = 10):
        super().__init__(name)
        self.max_positions = max_positions
    
    def validate(self, ctx: RiskContext, order: OrderIntent) -> Tuple[bool, Optional[str]]:
        if order.side != OrderSide.BUY:
            return True, None
        
        if len(ctx.account.positions) >= self.max_positions:
            return False, f"持仓数量已达上限 {self.max_positions}"
        
        return True, None
```

## 设计原则

1. **关注点分离**: OrderIntent 定义结构，RiskManager 定义接口，RiskContext 提供上下文
2. **管道模式**: 多个风险管理器可以组合使用，按顺序应用
3. **可扩展性**: 继承 RiskManager 可快速实现新的风险管理器
4. **灵活性**: 支持拒绝、调整、组合层面调整、交易后记录等多种能力

## 注意事项

1. 订单验证顺序：先 validate()，再 adjust()，最后 pre_trade()
2. 订单调整：adjust() 返回的订单应保持原始订单的基本信息
3. 组合层面调整：pre_trade() 可同时调整多个订单，用于组合层面的风险控制
4. 交易后记录：post_trade() 用于记录风控状态，不影响当前交易但影响后续交易
