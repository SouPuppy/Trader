# RAG 系统文档

## 概述

RAG（Retrieval-Augmented Generation）系统是一个统一的检索增强生成框架，用于基于多源数据（新闻、交易历史、趋势特征）回答金融相关问题。

## 架构

### 数据抽象

- **Document**: 统一文档对象，包含 `doc_id`, `doc_type`, `stock_code`, `timestamp`, `payload`, `text`
- **Candidate**: 检索候选，包含 `doc`, `recall_score`, `recall_source`
- **EvidenceItem**: 最终证据卡片，包含 `doc_id`, `doc_type`, `key_facts`, `signals`, `relevance_score`

### 文档类型

- `trends`: 趋势特征（从 `trends_features` 表）
- `news_piece`: 新闻片段（从 `raw_data.news` 字段解析）
- `trade_history`: 交易历史（从 `trade_history` 表）

### 任务类型

- `market_state`: 市场状态查询
- `trade_explain`: 交易历史解释
- `risk_check`: 风险检查
- `news_impact`: 新闻影响分析
- `strategy_suggest`: 策略建议

## 模块说明

### 1. Request Normalizer (`normalize.py`)

标准化用户输入为统一的 `RagRequest`。

### 2. Retrieval Planner (`planner.py`)

根据请求生成检索计划，决定：
- 检索哪些文档类型
- 时间窗口
- 召回数量（recall_k）和最终数量（final_k）
- 约束条件

### 3. Retrievers (`retrievers/`)

- **NewsRetriever**: 从 `raw_data` 表检索新闻
- **TradeRetriever**: 从 `trade_history` 表检索交易历史
- **TrendsRetriever**: 从 `trends_features` 表检索趋势特征（使用特征相似度）

### 4. Candidate Merger (`merger.py`)

合并三类候选，按 `doc_id` 去重。

### 5. Reranker (`rerank.py`)

重排候选，考虑：
- 相关性分数
- 时间衰减
- 质量分数
- 多样性（对 news_piece）

### 6. Evidence Builder (`evidence.py`)

将候选转换为证据项，提取：
- `key_facts`: 关键事实（1-3条）
- `signals`: 数值信号（sentiment/impact/ret_5d等）

### 7. Prompt Builder (`prompt.py`)

生成严格格式的 prompt，确保 LLM 只基于证据回答。

### 8. Generator (`generate.py`)

调用 DeepSeek API 生成答案。

### 9. Verifier (`verify.py`)

后验校验：
- 引用校验（doc_id 是否存在）
- 时间校验（文档时间 <= 决策时间）
- 置信度门控（证据总分阈值）

### 10. Observability (`observability.py`)

记录所有关键信息：
- 请求和计划
- 召回数量
- 重排分数分布
- 证据包
- 验证结果

## 使用方法

### 基本用法

```python
from trader.rag import rag_answer

# 简单调用
result = rag_answer(
    question="AAPL.O 最近30天的市场趋势如何？",
    stock_code="AAPL.O",
    decision_time="2024-01-15T00:00:00",
    frequency="1d"
)

print(result.answer)
print(f"验证状态: {'通过' if result.passed else '未通过'}")
```

### 完整示例

参见 `trader/debug/test_rag.py`

## 数据库表结构

### trends_features

存储趋势特征（需要从 `raw_data` 派生）：

```sql
CREATE TABLE trends_features (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  stock_code TEXT NOT NULL,
  bar_time TEXT NOT NULL,
  frequency TEXT NOT NULL,
  ret_1d REAL, ret_5d REAL, ret_20d REAL,
  range_pct REAL, gap_pct REAL, close_to_open REAL,
  vol_20d REAL, vol_60d REAL, vol_z_20d REAL,
  pe_ratio REAL, pe_ratio_ttm REAL,
  pcf_ratio_ttm REAL, pb_ratio REAL,
  ps_ratio REAL, ps_ratio_ttm REAL,
  UNIQUE(stock_code, bar_time, frequency)
);
```

### trade_history

存储交易历史：

```sql
CREATE TABLE trade_history (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  stock_code TEXT NOT NULL,
  trade_time TEXT NOT NULL,
  action TEXT NOT NULL,
  price REAL NOT NULL,
  volume REAL NOT NULL
);
```

### raw_data

新闻数据存储在 `raw_data.news` 字段（JSON 格式）。

## 注意事项

1. **数据库初始化**: 首次使用前需要确保数据库表存在（调用 `ensure_tables()`）
2. **新闻分析**: 新闻的 `sentiment` 和 `impact` 需要实时分析或从缓存获取
3. **趋势特征**: `trends_features` 表需要预先计算和填充
4. **API 密钥**: 需要配置 `DEEPSEEK_API_KEY` 环境变量

## 未来扩展

- 向量检索（使用 embedding）
- BM25 文本检索
- 更复杂的重排算法
- 缓存机制
- 批量处理

