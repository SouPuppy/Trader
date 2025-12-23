# Dataloader 模块

提供从开始日期到结束日期的所有 features 加载功能，支持五种不同的数据处理模式。

## 功能

- **dataloader_raw**: 原始数据加载器，不进行任何补全，节假日返回 None
- **dataloader_nullcomplete**: Null 补全数据加载器，自动插值补全交易日中的 null 值，但节假日仍返回 None
- **dataloader_autocomplete**: 自动插值补全数据加载器，自动插值补全所有 null 值，包括节假日
- **dataloader_ffill**: 前向填充数据加载器，使用前向填充补全所有 null 值，包括节假日
- **dataloader_linear**: 线性插值数据加载器，使用线性插值补全所有 null 值，包括节假日

## 使用方法

### 基本用法

```python
from trader.dataloader import (
    dataloader_raw,
    dataloader_nullcomplete,
    dataloader_autocomplete,
    dataloader_ffill,
    dataloader_linear
)

# 创建加载器
symbol = "AAPL.O"
raw_loader = dataloader_raw(symbol)
nullcomplete_loader = dataloader_nullcomplete(symbol)
autocomplete_loader = dataloader_autocomplete(symbol)
ffill_loader = dataloader_ffill(symbol)
linear_loader = dataloader_linear(symbol)

# 加载数据
start_date = "2023-01-01"
end_date = "2023-12-31"

# 原始数据（不补全）
raw_df = raw_loader.load(start_date, end_date)

# Null 补全（交易日中的 null 补全，节假日仍为 None）
nullcomplete_df = nullcomplete_loader.load(start_date, end_date)

# 自动补全（所有 null 补全，包括节假日）
autocomplete_df = autocomplete_loader.load(start_date, end_date)

# 前向填充（所有 null 补全，包括节假日）
ffill_df = ffill_loader.load(start_date, end_date)

# 线性插值（所有 null 补全，包括节假日）
linear_df = linear_loader.load(start_date, end_date)
```

### 指定特征

```python
# 只加载特定特征
feature_names = ["close_price", "volume", "ret_1d"]
df = raw_loader.load(start_date, end_date, feature_names=feature_names)
```

### 强制重新计算

```python
# 强制重新计算，忽略缓存
df = raw_loader.load(start_date, end_date, force=True)
```

## 输出格式

所有 Dataloader 返回的 DataFrame 格式相同：
- **索引**: 日期（DatetimeIndex），包含从开始日期到结束日期的所有日期（包括节假日）
- **列**: 特征名称
- **值**: 特征值
  - dataloader_raw: 节假日为 None，交易日可能有 None（如果特征计算失败）
  - dataloader_nullcomplete: 节假日为 None，交易日中的 None 会被插值补全
  - dataloader_autocomplete: 所有 None 都会被插值补全（包括节假日），使用前向填充+后向填充+线性插值
  - dataloader_ffill: 所有 None 都会被前向填充补全（包括节假日）
  - dataloader_linear: 所有 None 都会被线性插值补全（包括节假日）

## 节假日判断

节假日通过检查数据库中是否有该日期的数据来判断：
- 如果数据库中有该日期的数据 → 交易日
- 如果数据库中没有该日期的数据 → 节假日（返回 None）

## 插值方法

- **前向填充 (ffill)**: 使用前一个非空值填充
- **后向填充 (bfill)**: 使用后一个非空值填充
- **线性插值**: 使用线性插值方法填充缺失值

## 各 Dataloader 的区别

| Dataloader | 交易日 null 处理 | 节假日处理 | 插值方法 |
|------------|-----------------|-----------|---------|
| dataloader_raw | 保持 None | 保持 None | 无 |
| dataloader_nullcomplete | 插值补全 | 保持 None | ffill + bfill + linear |
| dataloader_autocomplete | 插值补全 | 插值补全 | ffill + bfill + linear |
| dataloader_ffill | 前向填充 | 前向填充 | ffill |
| dataloader_linear | 线性插值 | 线性插值 | linear |

## 测试和可视化

使用 `trader/debug/test_dataloader.py` 来比较所有 Dataloader 的输出：

```bash
python -m trader.debug.test_dataloader \
    --symbol AAPL.O \
    --start-date 2023-01-01 \
    --end-date 2023-12-31 \
    --features close_price volume ret_1d
```

这会生成：
1. 每个特征的对比图表（保存在 `output/dataloader_comparison/`），包含5个子图
2. 统计报告（显示每个特征的缺失值情况）

## 示例

```python
from trader.dataloader import dataloader_raw, dataloader_ffill, dataloader_linear

# 加载数据
symbol = "AAPL.O"
start_date = "2023-01-01"
end_date = "2023-12-31"

raw_loader = dataloader_raw(symbol)
raw_df = raw_loader.load(start_date, end_date)

# 查看数据
print(raw_df.head())
print(f"总日期数: {len(raw_df)}")
print(f"节假日数: {raw_df.isna().all(axis=1).sum()}")
print(f"交易日数: {(~raw_df.isna().all(axis=1)).sum()}")

# 查看特定特征
print(raw_df['close_price'].head(20))

# 使用前向填充
ffill_loader = dataloader_ffill(symbol)
ffill_df = ffill_loader.load(start_date, end_date)
print(f"前向填充后缺失值: {ffill_df['close_price'].isna().sum()}")

# 使用线性插值
linear_loader = dataloader_linear(symbol)
linear_df = linear_loader.load(start_date, end_date)
print(f"线性插值后缺失值: {linear_df['close_price'].isna().sum()}")
```
