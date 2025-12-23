# Chasing Extremes Agent 实验（带 Position Scaling 风险控制）

## 简介

这是 Chasing Extremes Agent 的带 Position Scaling 风险控制版本，用于测试 Position Scaling 风险控制是否有效。

## 策略逻辑（稳定亏钱的疯狂策略 - 反向操作）

- **反向操作**：价格上涨时卖出，价格下跌时买入（总是在错误的时间交易）
- **低阈值触发**：极端波动阈值降低到 1%，让它更容易触发交易
- **最小仓位保证**：即使波动很小，也至少给予 50% 仓位，让它频繁交易
- **Position Scaling 风险控制**：根据账户权益、置信度、历史表现动态调整仓位

## Position Scaling 风险控制

Position Scaling 风险管理器根据多种因素动态调整仓位大小：

### 1. 账户权益缩放
- **原理**：账户越大，单笔交易比例可能越小（风险分散）
- **公式**：`scaling = (base_equity / equity) ^ equity_scaling_factor`
- **参数**：
  - `base_equity`: 基准账户权益（默认 1,000,000）
  - `equity_scaling_factor`: 权益缩放因子（默认 0.5）

### 2. 置信度缩放
- **原理**：置信度越低，仓位越小
- **公式**：`scaling = confidence ^ confidence_power`
- **参数**：
  - `confidence_power`: 置信度幂次（默认 1.0，线性）
  - `confidence_power = 2.0`: 平方（更保守）

### 3. 历史表现缩放
- **原理**：连续亏损后降低仓位，连续盈利后增加仓位
- **参数**：
  - `consecutive_loss_threshold`: 连续亏损阈值（默认 3）
  - `consecutive_win_threshold`: 连续盈利阈值（默认 3）
  - `loss_scaling_factor`: 连续亏损后的缩放因子（默认 0.5）
  - `win_scaling_factor`: 连续盈利后的缩放因子（默认 1.2，上限 1.0）

### 4. 波动率缩放（可选）
- **原理**：波动率越高，仓位越小（Kelly Criterion 变种）
- **注意**：需要额外的波动率数据，默认关闭

## 综合缩放因子

所有缩放因子取**最小值**（最保守），确保最保守的风险控制。

## 参数说明

- `extreme_threshold`: 极端波动阈值（默认 1%，降低阈值让它更容易触发）
- `lookback_days`: 回看天数，用于计算涨跌幅（默认 1 天）
- `max_position_weight`: 最大仓位（默认 100%，全仓）
- `chase_up`: 是否追涨（默认 True，反向操作：价格上涨时卖出）
- `chase_down`: 是否追跌（默认 True，反向操作：价格下跌时买入）
- `enable_equity_scaling`: 是否启用账户权益缩放（默认 True）
- `enable_confidence_scaling`: 是否启用置信度缩放（默认 True）
- `enable_performance_scaling`: 是否启用历史表现缩放（默认 True）

## 运行

```bash
python main.py
```

## 预期行为

这个 agent 会：
1. **频繁交易**：由于阈值降低到 1%，它会频繁地触发交易
2. **反向操作**：价格上涨时卖出，价格下跌时买入（稳定亏钱）
3. **动态仓位调整**：根据账户权益、置信度、历史表现动态调整仓位
4. **风险控制**：Position Scaling 会降低高风险交易的仓位

## 用途

这个 agent 的主要用途是：
- 测试 Position Scaling 风险控制是否能够有效限制极端仓位
- 对比有 Position Scaling 和没有风险控制的策略表现
- 验证 Position Scaling 对策略收益的影响

## 与 1.3.0 和 1.3.1 的对比

- **1.3.0**: 没有风险控制，会全仓反向操作
- **1.3.1**: 有杠杆限制风险控制，限制总持仓市值不超过账户权益的 80%
- **1.3.2**: 有 Position Scaling 风险控制，根据账户权益、置信度、历史表现动态调整仓位

通过对比这三个实验的结果，可以评估不同风险控制策略的有效性。

