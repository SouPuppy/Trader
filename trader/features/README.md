# 特征系统文档

## 特征分类

### 1. 基础列（8个）
- `date`, `stock_code`, `close_price`, `volume`
- `prev_close`, `open_price`, `high_price`, `low_price`

### 2. 收益率特征（3个）

| 特征 | 公式 | Lookback |
|------|------|----------|
| `ret_1d` | `close_price / prev_close - 1` | 1天 |
| `ret_5d` | `close_price / close_price.shift(5) - 1` | 5天 |
| `ret_20d` | `close_price / close_price.shift(20) - 1` | 20天 |

### 3. 日内特征（3个）

| 特征 | 公式 | Lookback | 说明 |
|------|------|----------|------|
| `range_pct` | `(high_price - low_price) / prev_close` | 1天 | 日内波动率 |
| `gap_pct` | `(open_price - prev_close) / prev_close` | 1天 | 跳空幅度 |
| `close_to_open` | `close_price / open_price - 1` | 0天 | 收盘相对开盘 |

### 4. 波动率特征（3个）

| 特征 | 公式 | Lookback |
|------|------|----------|
| `vol_20d` | rolling 20日 `ret_1d` 标准差 | 20天 |
| `vol_60d` | rolling 60日 `ret_1d` 标准差 | 60天 |
| `vol_z_20d` | `(volume - volume_20d_mean) / volume_20d_std` | 20天 |

### 5. 财务比率（6个）
- `pe_ratio`, `pe_ratio_ttm`, `pcf_ratio_ttm`
- `pb_ratio`, `ps_ratio`, `ps_ratio_ttm`

**总计: 23个特征**

## 使用

```bash
# 查看所有特征
python -m trader.cmd.build_features --list

# 计算特征
python -m trader.cmd.build_features --date 2023-01-03 --symbol AAPL.O --feature ret_1d

# 生成图表
python -m trader.features.visualize.daily_features --output ./output/features
```

## 注意事项

- 缺失值：使用前向填充（forward fill）
- 异常值：自动处理无穷大值
- Lookback：系统自动加载所需历史数据

## 待实现

- 新闻特征：`news_count`, `sent_mean`, `impact_sum`, `impact_weighted_sent`, `top_topics_json`
