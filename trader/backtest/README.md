# 回测报告说明

本文档说明回测报告中包含的所有数据及其含义。

## 报告文件

回测完成后会生成以下文件：
- **文本报告** (`backtest_report_{stock_code}_{start_date}_{end_date}.txt`): 人类可读的文本格式报告
- **JSON 报告** (`backtest_report_{stock_code}_{start_date}_{end_date}.json`): 结构化的 JSON 格式报告，便于程序处理
- **图表文件** (`backtest_charts_{stock_code}_{start_date}_{end_date}.png`): 可视化图表

## 报告数据结构

### 1. 基本信息

| 字段 | 说明 |
|------|------|
| `stock_code` | 股票代码，如 "AAPL.O" |
| `start_date` | 回测开始日期，格式：YYYY-MM-DD |
| `end_date` | 回测结束日期，格式：YYYY-MM-DD |
| `trading_days` | 交易日总数（不包括周末和节假日） |

### 2. 账户摘要

| 字段 | 说明 |
|------|------|
| `initial_cash` | 初始资金（元），回测开始时的现金 |
| `cash` | 最终现金（元），回测结束时的可用现金 |
| `positions_value` | 最终持仓市值（元），所有持仓按当前价格计算的总市值 |
| `equity` | 最终总权益（元），等于 `cash + positions_value` |
| `profit` | 总盈亏（元），等于 `equity - initial_cash` |
| `return_pct` | 总收益率（%），等于 `(profit / initial_cash) * 100` |

### 3. 交易记录 (trades)

每笔交易记录包含以下字段：

| 字段 | 说明 |
|------|------|
| `date` | 交易日期时间 |
| `type` | 交易类型：`"buy"`（买入）或 `"sell"`（卖出） |
| `stock_code` | 股票代码 |
| `shares` | 交易股数 |
| `price` | 交易价格（元/股） |
| `cost` | 买入成本（元），仅买入交易有此字段，等于 `shares * price` |
| `revenue` | 卖出收入（元），仅卖出交易有此字段，等于 `shares * price` |
| `profit` | 交易利润（元），仅卖出交易有此字段，等于 `revenue - (shares * average_price)` |

### 4. 最终持仓 (positions)

回测结束时的持仓明细，每个持仓包含：

| 字段 | 说明 |
|------|------|
| `shares` | 持仓股数 |
| `average_price` | 平均成本价（元/股），多次买入时按加权平均计算 |
| `current_price` | 当前价格（元/股），回测结束时的市场价格 |
| `market_value` | 持仓市值（元），等于 `shares * current_price` |
| `profit` | 持仓盈亏（元），等于 `(current_price - average_price) * shares` |

### 5. 统计信息 (statistics)

| 字段 | 说明 |
|------|------|
| `max_equity` | 最高权益（元），回测期间账户总权益的最高值 |
| `min_equity` | 最低权益（元），回测期间账户总权益的最低值 |
| `max_return_pct` | 最高收益率（%），回测期间总收益率的最高值 |
| `min_return_pct` | 最低收益率（%），回测期间总收益率的最低值 |
| `max_drawdown_pct` | 最大回撤（%），从最高点到最低点的最大跌幅 |

**最大回撤计算方式**：
- 从回测开始，记录每个时间点的权益值
- 对于每个时间点，计算从历史最高点到当前点的回撤：`(peak - current) / peak * 100`
- 取所有回撤值中的最大值

### 6. 夏普比率 (sharpe_ratio)

夏普比率用于衡量投资组合的风险调整后收益。

| 字段 | 说明 |
|------|------|
| `sharpe_daily` | 日频夏普比率，基于日收益率计算 |
| `sharpe_annual` | 年化夏普比率，等于 `sqrt(252) * sharpe_daily` |
| `risk_free_rate_annual` | 年化无风险利率（默认 0.0） |
| `risk_free_rate_daily` | 日化无风险利率，等于 `(1 + R_f(ann))^(1/252) - 1` |
| `mean_excess_return` | 超额收益均值，等于日收益率均值减去无风险利率 |
| `std_excess_return` | 超额收益标准差 |
| `num_trading_days` | 用于计算的交易日数 |
| `daily_returns` | 日收益率序列，`r_t = E_t / E_{t-1} - 1`，其中 `E_t` 是第 t 天的账户净值 |
| `excess_returns` | 超额日收益序列，`x_t = r_t - r_f` |

**夏普比率计算公式**：
- 日频夏普比率：`Sharpe_daily = mean_excess_return / std_excess_return`
- 年化夏普比率：`Sharpe_annual = sqrt(252) * Sharpe_daily`

**说明**：
- 夏普比率越高，表示风险调整后的收益越好
- 通常认为：
  - 夏普比率 < 1：表现较差
  - 1 ≤ 夏普比率 < 2：表现良好
  - 2 ≤ 夏普比率 < 3：表现优秀
  - 夏普比率 ≥ 3：表现卓越

### 7. 每日记录 (daily_records)

回测期间每个交易日的账户状态快照，包含：

| 字段 | 说明 |
|------|------|
| `date` | 日期 |
| `cash` | 当日现金（元） |
| `positions_value` | 当日持仓市值（元） |
| `equity` | 当日总权益（元） |
| `profit` | 当日累计盈亏（元） |
| `return_pct` | 当日累计收益率（%） |
| `positions` | 当日持仓明细（格式同"最终持仓"） |

## 使用建议

1. **评估策略表现**：重点关注总收益率、最大回撤和夏普比率
2. **分析交易行为**：查看交易记录，了解买入卖出时机和频率
3. **风险控制**：通过最大回撤和每日记录，识别策略的风险暴露期
4. **资金管理**：通过现金和持仓市值数据，评估资金利用效率
5. **策略优化**：结合每日记录，分析策略在不同市场环境下的表现

## 注意事项

- 所有金额单位均为人民币（元）
- 收益率和回撤均为百分比（%）
- 日期格式为 YYYY-MM-DD
- 持仓的平均成本价采用加权平均法计算
- 夏普比率计算假设一年有 252 个交易日

