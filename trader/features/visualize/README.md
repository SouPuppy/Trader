# 特征可视化工具

基于 matplotlib 的特征可视化工具，生成静态图表文件来查看和分析股票特征走势。

## 功能特性

- **高质量图表** - 使用 matplotlib 生成清晰的 PNG 图表
- **多股票对比** - 在同一图表中对比多个股票的特征走势
- **多特征分析** - 支持生成所有特征的图表
- **批量生成** - 一次性生成所有特征的图表
- **灵活筛选** - 可选择特定的股票和特征进行可视化

## 快速开始

### 1. 生成所有特征的图表

```bash
# 生成所有股票的所有特征图表
python -m trader.features.visualize.daily_features --output ./output/features

# 生成指定股票的特征图表
python -m trader.features.visualize.daily_features --symbols AAPL.O MSFT.O --output ./output/features

# 使用脚本
./script/A. visualize features.sh ./output/features
```

### 2. 查看图表

生成的图表会保存在指定的输出目录中，每个特征一个 PNG 文件：
- `pe_ratio.png`
- `pe_ratio_ttm.png`
- `close_price.png`
- `volume.png`
- 等等...

## 命令行选项

```bash
python -m trader.features.visualize.daily_features [选项]

选项:
  --symbols SYMBOL1 SYMBOL2 ...  指定股票代码列表（可选，不指定则使用所有股票）
  --output OUTPUT_DIR             输出目录（可选，不指定则显示图表）
  --figsize WIDTH HEIGHT          图表大小（默认: 14 7）
  --dpi DPI                       图表分辨率（默认: 150）
  --no-summary                    不打印特征汇总信息
  --force                         强制重新计算，忽略缓存
```

## 使用示例

### 示例 1: 生成所有特征的图表

```bash
python -m trader.features.visualize.daily_features --output ./output/features
```

### 示例 2: 只生成指定股票的特征

```bash
python -m trader.features.visualize.daily_features \
    --symbols AAPL.O MSFT.O GOOGL.O \
    --output ./output/selected_features
```

### 示例 3: 自定义图表大小和分辨率

```bash
python -m trader.features.visualize.daily_features \
    --output ./output/features \
    --figsize 16 8 \
    --dpi 200
```

### 示例 4: 查看特征列表

```bash
python -m trader.cmd.build_features --list
```

## 图表特性

每个生成的图表包含：

- **多股票对比**: 不同颜色和线型区分不同股票
- **清晰的图例**: 显示所有股票代码
- **统计信息**: 显示 symbols 数量和数据点数
- **日期格式化**: 自动调整日期刻度密度
- **高质量输出**: 支持自定义 DPI 和尺寸

## 技术栈

- **数据处理**: Pandas
- **图表生成**: Matplotlib
- **数据库**: SQLite3

## 注意事项

- 确保数据库文件 (`data/data.sqlite3`) 存在且包含数据
- 生成大量图表可能需要一些时间
- 图表文件会保存在指定的输出目录中

## 故障排除

### 数据库连接错误
确保数据库文件路径正确，检查 `trader/config.py` 中的 `DB_PATH` 设置。

### 图表不显示
检查输出目录是否有写入权限。

### 内存不足
如果股票或特征数量很大，可以分批生成：
```bash
# 先处理前几个股票
python -m trader.features.visualize.daily_features --symbols AAPL.O MSFT.O --output ./output/batch1

# 再处理其他股票
python -m trader.features.visualize.daily_features --symbols GOOGL.O TSLA.O --output ./output/batch2
```

