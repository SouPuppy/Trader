# RAG Agent Backtest

基于 RAG 系统的交易代理，使用 RAG 系统回答问题并做出交易决策。

## 特性

1. **RAG 集成**: 使用 RAG 系统分析股票市场状态和趋势
2. **交易限制**: 12 个月内最多交易 10 次，防止 LLM 过度交易
3. **交易记录**: 所有交易记录到 `trade_history` 表，可用于 RAG 查询
4. **自动清理**: 运行开始时自动清空 `trade_history` 表
5. **报告生成**: 自动生成报告到 `output/rag_agent/{timestamp}/report`

## 使用方法

```bash
python experiments/4.1\ RAG\ Agent/main.py
```

## 配置

在 `main.py` 中可以修改以下配置：

- `stock_codes`: 要交易的股票代码列表
- `start_date`: 回测开始日期 (YYYY-MM-DD)
- `end_date`: 回测结束日期 (YYYY-MM-DD)
- `initial_cash`: 初始资金
- `max_trades_per_12months`: 12 个月内最大交易次数（默认 10）
- `clear_trade_history`: 是否在开始时清空交易历史（默认 True）

## 工作原理

1. **RAG 查询**: Agent 使用 RAG 系统查询股票的市场状态和趋势
2. **评分**: 根据 RAG 的回答计算股票评分（-1 到 1）
3. **交易决策**:
   - 如果评分 < -0.3 且持有该股票，则卖出
   - 如果评分 > 0.3 且未持有，则买入
4. **交易限制**: 检查过去 12 个月的交易次数，如果达到上限则停止交易
5. **记录交易**: 所有交易记录到 `trade_history` 表

## 输出

报告保存在 `output/rag_agent/{timestamp}/` 目录下，包含：
- HTML 报告
- 交易记录
- 账户状态变化

## 注意事项

- 需要设置 `DEEPSEEK_API_KEY` 环境变量
- RAG 查询可能需要一些时间，请耐心等待
- 交易限制是为了防止 LLM 过度交易，可以根据需要调整

