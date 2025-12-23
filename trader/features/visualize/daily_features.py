"""
每日特征可视化
绘制所有 symbols 的特征走势图
"""
import sqlite3
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path
import sys
from typing import List, Optional, Dict
from datetime import datetime

try:
    from tqdm import tqdm
except ImportError:
    # 如果 tqdm 未安装，使用一个简单的替代实现
    def tqdm(iterable, desc=None, total=None, **kwargs):
        if desc:
            print(f"{desc}...")
        return iterable

# 添加项目根目录到 Python 路径
project_root = Path(__file__).parent.parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from trader.config import DB_PATH
from trader.logger import get_logger
from trader.features import get_feature_names, get_feature
from trader.features import features  # 导入特征定义以触发注册
from trader.features.cache import (
    get_cached_features_batch, 
    cache_features_batch, 
    ensure_features_table
)

logger = get_logger(__name__)


def get_all_symbols() -> List[str]:
    """
    从数据库获取所有唯一的股票代码
    
    Returns:
        股票代码列表
    """
    if not DB_PATH.exists():
        logger.error(f"数据库文件不存在: {DB_PATH}")
        return []
    
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        cursor.execute("SELECT DISTINCT stock_code FROM raw_data ORDER BY stock_code")
        symbols = [row[0] for row in cursor.fetchall()]
        
        conn.close()
        logger.info(f"找到 {len(symbols)} 个股票代码")
        return symbols
        
    except Exception as e:
        logger.error(f"获取股票代码时出错: {e}", exc_info=True)
        return []


def load_all_stock_data(symbols: Optional[List[str]] = None) -> pd.DataFrame:
    """
    从数据库加载所有股票的历史数据
    
    Args:
        symbols: 股票代码列表，如果为 None 则加载所有股票
        
    Returns:
        DataFrame，包含所有股票的历史数据
    """
    if not DB_PATH.exists():
        logger.error(f"数据库文件不存在: {DB_PATH}")
        return pd.DataFrame()
    
    try:
        conn = sqlite3.connect(DB_PATH)
        
        if symbols:
            # 使用参数化查询，避免 SQL 注入
            placeholders = ','.join(['?'] * len(symbols))
            query = f"""
                SELECT 
                    datetime,
                    stock_code,
                    prev_close,
                    open_price,
                    high_price,
                    low_price,
                    close_price,
                    volume,
                    pe_ratio,
                    pe_ratio_ttm,
                    pcf_ratio_ttm,
                    pb_ratio,
                    ps_ratio,
                    ps_ratio_ttm
                FROM raw_data 
                WHERE stock_code IN ({placeholders})
                ORDER BY stock_code, datetime
            """
            df = pd.read_sql_query(query, conn, params=tuple(symbols))
        else:
            query = """
                SELECT 
                    datetime,
                    stock_code,
                    prev_close,
                    open_price,
                    high_price,
                    low_price,
                    close_price,
                    volume,
                    pe_ratio,
                    pe_ratio_ttm,
                    pcf_ratio_ttm,
                    pb_ratio,
                    ps_ratio,
                    ps_ratio_ttm
                FROM raw_data 
                ORDER BY stock_code, datetime
            """
            df = pd.read_sql_query(query, conn)
        
        conn.close()
        
        if df.empty:
            logger.warning("数据库中没有数据")
            return pd.DataFrame()
        
        # 确保 datetime 是 datetime 类型
        if 'datetime' in df.columns:
            df['datetime'] = pd.to_datetime(df['datetime'])
        
        logger.info(f"加载了 {len(df)} 条记录，包含 {df['stock_code'].nunique()} 个股票")
        return df
        
    except Exception as e:
        logger.error(f"加载股票数据时出错: {e}", exc_info=True)
        return pd.DataFrame()


def compute_features_for_all_symbols(df: pd.DataFrame, force: bool = False) -> Dict[str, pd.DataFrame]:
    """
    为所有 symbols 计算所有特征
    
    Args:
        df: 包含所有股票数据的 DataFrame
        force: 是否强制重新计算，忽略缓存
        
    Returns:
        字典，key 为特征名称，value 为包含所有 symbols 该特征值的 DataFrame
        DataFrame 的列为 symbols，行为日期
    """
    # 确保 features 表存在
    ensure_features_table()
    
    feature_names = get_feature_names()
    results = {}
    
    symbols = df['stock_code'].unique()
    logger.info(f"开始计算特征: {len(symbols)} 个股票, {len(feature_names)} 个特征, force={force}")
    
    # 首先确定所有股票的日期范围（从最早到最晚）
    all_dates_set = set()
    for symbol in symbols:
        symbol_df = df[df['stock_code'] == symbol].copy()
        if 'datetime' in symbol_df.columns:
            symbol_df['datetime'] = pd.to_datetime(symbol_df['datetime'])
            all_dates_set.update(symbol_df['datetime'].unique())
        elif isinstance(symbol_df.index, pd.DatetimeIndex):
            all_dates_set.update(symbol_df.index.unique())
    
    if not all_dates_set:
        logger.warning("没有找到任何日期数据")
        return {}
    
    # 创建完整的日期索引（从最早到最晚，包含所有日期）
    min_date = min(all_dates_set)
    max_date = max(all_dates_set)
    full_date_range = pd.date_range(start=min_date, end=max_date, freq='D')
    logger.info(f"日期范围: {min_date.date()} 到 {max_date.date()}, 共 {len(full_date_range)} 天")
    
    # 按 symbol 分组处理
    for symbol in tqdm(symbols, desc="处理股票", unit="股票"):
        logger.debug(f"处理股票: {symbol}")
        symbol_df = df[df['stock_code'] == symbol].copy()
        
        # 确保 datetime 是索引
        if 'datetime' in symbol_df.columns:
            symbol_df = symbol_df.set_index('datetime').sort_index()
        elif not isinstance(symbol_df.index, pd.DatetimeIndex):
            logger.warning(f"无法为 {symbol} 设置 datetime 索引")
            continue
        
        # 重新索引，确保包含所有日期（缺失的日期会填充为 NaN）
        symbol_df = symbol_df.reindex(full_date_range)
        
        # 获取所有日期（包括缺失的日期）
        dates = [pd.Timestamp(dt).strftime('%Y-%m-%d') for dt in symbol_df.index]
        
        # 批量从缓存获取（如果 force=False）
        cached_data = {}
        if not force:
            cached_data = get_cached_features_batch(symbol, dates, feature_names)
        
        # 收集需要缓存的数据
        features_to_cache = []
        
        # 为每个特征计算值
        for feature_name in tqdm(feature_names, desc=f"计算 {symbol} 的特征", leave=False, unit="特征"):
            feature_spec = get_feature(feature_name)
            if feature_spec is None:
                logger.debug(f"跳过未找到的特征: {feature_name}")
                continue
            
            try:
                # 检查缓存中是否有该特征的所有日期数据
                need_compute = force
                if not need_compute:
                    # 检查该特征是否有缺失的日期
                    missing_dates = []
                    for date in dates:
                        if date not in cached_data or feature_name not in cached_data[date]:
                            missing_dates.append(date)
                    if missing_dates:
                        need_compute = True
                        logger.debug(f"特征 {feature_name} for {symbol}: 缓存中缺少 {len(missing_dates)} 个日期")
                
                if need_compute:
                    logger.debug(f"计算特征: {feature_name} for {symbol}")
                    # 计算特征
                    feature_series = feature_spec.compute(symbol_df)
                else:
                    # 从缓存构建 Series
                    logger.debug(f"从缓存加载特征: {feature_name} for {symbol}")
                    feature_dict = {}
                    for date in dates:
                        if date in cached_data and feature_name in cached_data[date]:
                            feature_dict[pd.Timestamp(date)] = cached_data[date][feature_name]
                    feature_series = pd.Series(feature_dict)
                
                # 初始化结果 DataFrame（如果还没有）
                if feature_name not in results:
                    results[feature_name] = pd.DataFrame()
                
                # 将结果添加到 DataFrame
                # 即使 feature_series 为空，也要添加列（可能所有值都是 NA）
                # 这样绘图时可以看到该股票存在，只是没有数据
                if results[feature_name].empty:
                    # 如果结果 DataFrame 为空，使用 feature_series 的索引创建
                    if not feature_series.empty:
                        results[feature_name] = pd.DataFrame(index=feature_series.index)
                    else:
                        # 如果 feature_series 也为空，使用 symbol_df 的索引
                        results[feature_name] = pd.DataFrame(index=symbol_df.index)
                
                # 添加该股票的特征值（即使全为 NA）
                # 使用 reindex 对齐索引，fill_value=None 表示缺失值填充为 NA
                aligned_series = feature_series.reindex(results[feature_name].index, fill_value=None)
                results[feature_name][symbol] = aligned_series
                
                original_non_null = feature_series.notna().sum()
                aligned_non_null = aligned_series.notna().sum()
                logger.debug(f"特征 {feature_name} for {symbol}: 计算完成, "
                           f"原始: {len(feature_series)} 个值, {original_non_null} 个非空; "
                           f"对齐后: {len(aligned_series)} 个值, {aligned_non_null} 个非空")
                
                # 收集需要缓存的数据（只缓存非 NA 的值）
                if need_compute:
                    for dt, value in feature_series.items():
                        if pd.notna(value):  # 只缓存非 NA 的值
                            date_str = pd.Timestamp(dt).strftime('%Y-%m-%d')
                            features_to_cache.append((symbol, date_str, feature_name, value))
                    
            except Exception as e:
                logger.warning(f"计算特征 {feature_name} 对于 {symbol} 时出错: {e}", exc_info=True)
                continue
        
        # 批量缓存
        if features_to_cache:
            logger.debug(f"批量缓存 {len(features_to_cache)} 个特征值 for {symbol}")
            cache_features_batch(features_to_cache)
    
    # 确保所有 DataFrame 的索引都是 datetime，并且对齐
    logger.debug("对齐所有特征 DataFrame 的索引")
    
    # 使用之前确定的完整日期范围作为统一索引
    # 这样确保所有特征 DataFrame 都包含从开始到结束的所有日期
    unified_index = full_date_range
    logger.debug(f"统一索引: {len(unified_index)} 个日期（从 {unified_index[0].date()} 到 {unified_index[-1].date()}）")
    
    # 为每个特征 DataFrame 统一索引
    for feature_name in results:
        if not results[feature_name].empty:
            # 记录统一索引前的数据统计
            before_data = {}
            for col in results[feature_name].columns:
                col_data = results[feature_name][col].dropna()
                if len(col_data) > 0:
                    before_data[col] = {
                        'count': len(col_data),
                        'min': col_data.min(),
                        'max': col_data.max(),
                        'mean': col_data.mean(),
                        'unique': col_data.nunique()
                    }
            
            results[feature_name].index = pd.to_datetime(results[feature_name].index)
            # 重新索引，确保所有股票使用相同的日期索引
            # 使用 method=None 确保不进行插值，只对齐索引
            results[feature_name] = results[feature_name].reindex(unified_index, method=None).sort_index()
            
            # 记录统一索引后的数据统计
            logger.debug(f"特征 {feature_name}: 统一索引后, {len(results[feature_name].columns)} 个股票")
            for col in results[feature_name].columns:
                col_data = results[feature_name][col].dropna()
                if len(col_data) > 0:
                    after_info = {
                        'count': len(col_data),
                        'min': col_data.min(),
                        'max': col_data.max(),
                        'mean': col_data.mean(),
                        'unique': col_data.nunique()
                    }
                    before_info = before_data.get(col, {})
                    if before_info.get('count', 0) != after_info['count']:
                        logger.warning(f"  {col}: 数据点数量变化 {before_info.get('count', 0)} -> {after_info['count']}")
                    if before_info.get('unique', 0) != after_info['unique']:
                        logger.warning(f"  {col}: 唯一值数量变化 {before_info.get('unique', 0)} -> {after_info['unique']}")
                    logger.debug(f"  {col}: {after_info['count']} 个数据点, "
                               f"值范围: {after_info['min']:.4f} 到 {after_info['max']:.4f}, "
                               f"{after_info['unique']} 个唯一值")
        else:
            # 如果为空，使用统一索引创建
            results[feature_name] = pd.DataFrame(index=unified_index)
    
    logger.info(f"特征计算完成: 共 {len(results)} 个特征")
    return results


def plot_feature_trends(
    feature_data: Dict[str, pd.DataFrame],
    output_dir: Optional[Path] = None,
    figsize: tuple = (14, 7),
    dpi: int = 150
):
    """
    绘制所有特征的趋势图
    
    Args:
        feature_data: 特征数据字典，key 为特征名称，value 为 DataFrame
        output_dir: 输出目录，如果为 None 则显示图表
        figsize: 图表大小
        dpi: 图表分辨率
    """
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
    
    # 设置中文字体（如果需要）
    plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    
    # 定义更多样化的样式组合
    markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*', 'h', 'H', 'X']
    linestyles = ['-', '--', '-.', ':', (0, (3, 1, 1, 1)), (0, (5, 5))]
    
    # 使用更丰富的颜色方案
    def get_color_palette(n):
        """根据数量选择合适的颜色方案"""
        if n <= 10:
            return plt.cm.tab10(range(n))
        elif n <= 20:
            return plt.cm.tab20(range(n))
        else:
            # 使用 Set3 或其他更丰富的调色板
            return plt.cm.Set3(range(n))
    
    plotted_count = 0
    skipped_count = 0
    
    for feature_name, df in feature_data.items():
        if df.empty:
            logger.warning(f"特征 {feature_name} 没有数据，跳过")
            skipped_count += 1
            continue
        
        # 检查是否有有效数据
        valid_symbols = [s for s in df.columns if df[s].notna().sum() > 0]
        if not valid_symbols:
            logger.warning(f"特征 {feature_name} 没有有效数据，跳过")
            skipped_count += 1
            continue
        
        # 调试信息：显示每个股票的数据统计，并检查数据是否不同
        logger.debug(f"特征 {feature_name} 的数据统计:")
        symbol_data_samples = {}
        for symbol in df.columns:
            non_null_count = df[symbol].notna().sum()
            total_count = len(df[symbol])
            valid_data = df[symbol].dropna()
            if len(valid_data) > 0:
                symbol_data_samples[symbol] = {
                    'count': len(valid_data),
                    'min': valid_data.min(),
                    'max': valid_data.max(),
                    'mean': valid_data.mean(),
                    'first_5': valid_data.head(5).tolist()
                }
            logger.debug(f"  {symbol}: {non_null_count}/{total_count} 非空值")
        
        # 检查是否有多个股票的数据完全相同
        if len(symbol_data_samples) > 1:
            data_signatures = {}
            for symbol, data_info in symbol_data_samples.items():
                # 使用前5个值和统计信息作为签名
                sig = tuple(data_info['first_5'] + [data_info['mean'], data_info['min'], data_info['max']])
                if sig not in data_signatures:
                    data_signatures[sig] = []
                data_signatures[sig].append(symbol)
            
            if len(data_signatures) < len(symbol_data_samples):
                logger.warning(f"警告: 发现 {len(symbol_data_samples) - len(data_signatures)} 组股票的数据可能相同!")
                for sig, symbols in data_signatures.items():
                    if len(symbols) > 1:
                        logger.warning(f"  以下股票的数据可能相同: {symbols}")
        
        logger.info(f"绘制特征 {feature_name} 的趋势图（{len(valid_symbols)} 个symbols）...")
        
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
        
        # 获取颜色方案
        colors = get_color_palette(len(valid_symbols))
        max_data_points = 0
        
        # 为每个 symbol 绘制一条线，使用不同的样式组合
        plotted_lines = 0
        for idx, symbol in enumerate(valid_symbols):
            if df[symbol].notna().sum() == 0:
                logger.debug(f"跳过 {symbol}：没有非空值")
                continue
            
            # 获取该股票的所有数据（包括 NaN）
            all_data = df[symbol]
            
            # 检查是否有有效数据
            valid_data = all_data.dropna()
            if len(valid_data) == 0:
                logger.debug(f"跳过 {symbol}：dropna 后没有数据")
                continue
            
            # 检查数据是否唯一（避免所有股票数据相同）
            unique_values = valid_data.nunique()
            if unique_values == 1 and len(valid_data) > 1:
                logger.debug(f"警告: {symbol} 的所有数据点都相同: {valid_data.iloc[0]}")
            
            max_data_points = max(max_data_points, len(valid_data))
            
            # 根据数据点数量决定是否显示标记
            marker_style = markers[idx % len(markers)] if len(valid_data) <= 50 else None
            marker_size = 4 if len(valid_data) <= 30 else 2
            
            # 使用不同的线型和颜色
            line_style = linestyles[idx % len(linestyles)]
            line_width = 2.0 if idx < 5 else 1.5  # 前5个symbols用更粗的线
            
            logger.debug(f"绘制 {symbol}: {len(valid_data)} 个数据点, 值范围: {valid_data.min():.4f} 到 {valid_data.max():.4f}")
            
            # 使用 numpy.ma.masked_array 在缺失数据处断开连线
            import numpy as np
            import numpy.ma as ma
            
            # 创建 masked array，NaN 值会被 mask
            masked_values = ma.masked_invalid(all_data.values)
            masked_index = all_data.index
            
            # 绘制（masked array 会自动在缺失数据处断开）
            ax.plot(
                masked_index, 
                masked_values, 
                label=symbol, 
                marker=marker_style,
                markersize=marker_size,
                markevery=max(1, len(valid_data) // 20),  # 每隔一定距离显示一个标记
                linestyle=line_style,
                linewidth=line_width,
                color=colors[idx],
                alpha=0.8,
                zorder=len(valid_symbols) - idx  # 后面的symbols在上层
            )
            plotted_lines += 1
        
        logger.info(f"实际绘制了 {plotted_lines} 条线（共 {len(valid_symbols)} 个有效股票）")
        
        # 设置图表属性
        ax.set_title(f'{feature_name} 趋势图', fontsize=16, fontweight='bold', pad=15)
        ax.set_xlabel('日期', fontsize=13, fontweight='bold')
        ax.set_ylabel(feature_name, fontsize=13, fontweight='bold')
        ax.grid(True, alpha=0.4, linestyle='--', linewidth=0.8)
        
        # 改进图例设置
        if len(valid_symbols) <= 8:
            ax.legend(loc='best', fontsize=10, framealpha=0.9, edgecolor='gray')
        elif len(valid_symbols) <= 15:
            ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=8, 
                     framealpha=0.9, edgecolor='gray', ncol=1)
        else:
            ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=7, 
                     framealpha=0.9, edgecolor='gray', ncol=2)
        
        # 格式化 x 轴日期
        if max_data_points > 0:
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
            # 根据数据范围自动调整日期刻度
            if max_data_points > 60:
                ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=4))
            elif max_data_points > 30:
                ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=2))
            elif max_data_points > 10:
                ax.xaxis.set_major_locator(mdates.WeekdayLocator())
            else:
                ax.xaxis.set_major_locator(mdates.DayLocator())
            
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        # 添加统计信息文本框
        stats_text = f"Symbols: {len(valid_symbols)} | Data Points: {max_data_points}"
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
               fontsize=9, verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        
        if output_dir:
            output_file = output_dir / f'{feature_name}.png'
            plt.savefig(output_file, dpi=dpi, bbox_inches='tight', facecolor='white')
            logger.info(f"✓ 保存图表到: {output_file}")
            plt.close()
            plotted_count += 1
        else:
            plt.show()
            plotted_count += 1
    
    logger.info(f"图表绘制完成: 成功 {plotted_count} 个，跳过 {skipped_count} 个")


def print_feature_summary(feature_data: Dict[str, pd.DataFrame], output_file: Optional[Path] = None):
    """
    打印特征汇总信息
    
    Args:
        feature_data: 特征数据字典
        output_file: 输出文件路径（可选）
    """
    all_feature_names = get_feature_names()
    
    summary_lines = []
    summary_lines.append("=" * 80)
    summary_lines.append("特征汇总报告")
    summary_lines.append("=" * 80)
    summary_lines.append(f"\n总特征数: {len(all_feature_names)}")
    summary_lines.append(f"成功计算的特征数: {len(feature_data)}")
    summary_lines.append(f"缺失的特征数: {len(all_feature_names) - len(feature_data)}")
    summary_lines.append("\n" + "-" * 80)
    summary_lines.append("特征详情:")
    summary_lines.append("-" * 80)
    
    for feature_name in all_feature_names:
        if feature_name in feature_data:
            df = feature_data[feature_name]
            valid_symbols = [s for s in df.columns if df[s].notna().sum() > 0]
            total_points = sum(df[s].notna().sum() for s in df.columns)
            summary_lines.append(
                f"  ✓ {feature_name:20s}\t| Symbols: {len(valid_symbols):3d}\t| "
                f"Data Points: {total_points:5d}\t| Status: OK"
            )
        else:
            summary_lines.append(
                f"  ✗ {feature_name:20s}\t| Symbols:    0\t| "
                f"Data Points:     0\t| Status: NO DATA"
            )
    
    summary_lines.append("\n" + "=" * 80)
    
    summary_text = "\n".join(summary_lines)
    
    # 打印到控制台
    print("\n" + summary_text)
    logger.info("特征汇总信息已输出")
    
    # 保存到文件
    if output_file:
        output_file = Path(output_file)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(summary_text)
        logger.info(f"特征汇总已保存到: {output_file}")


def plot_all_features(
    symbols: Optional[List[str]] = None,
    output_dir: Optional[Path] = None,
    figsize: tuple = (14, 7),
    dpi: int = 150,
    print_summary: bool = True,
    force: bool = False
):
    """
    绘制所有 symbols 的所有特征走势图
    
    Args:
        symbols: 股票代码列表，如果为 None 则使用所有股票
        output_dir: 输出目录，如果为 None 则显示图表
        figsize: 图表大小
        dpi: 图表分辨率
        print_summary: 是否打印特征汇总信息
    """
    from trader.logger import log_separator, log_section
    for line in log_section("开始特征可视化流程", width=80):
        logger.info(line)
    
    # 如果没有指定 symbols，获取所有 symbols
    if symbols is None:
        symbols = get_all_symbols()
        if not symbols:
            logger.error("没有找到任何股票代码")
            return
    
    logger.info(f"处理 {len(symbols)} 个股票代码: {symbols}")
    
    # 加载数据
    logger.info("加载股票数据...")
    df = load_all_stock_data(symbols)
    if df.empty:
        logger.error("没有加载到任何数据")
        return
    
    # 获取所有应该计算的特征
    all_feature_names = get_feature_names()
    logger.info(f"需要计算的特征列表 ({len(all_feature_names)} 个):")
    for i, name in enumerate(all_feature_names, 1):
        logger.info(f"  {i:2d}. {name}")
    
    # 计算特征
    logger.info("\n开始计算特征...")
    feature_data = compute_features_for_all_symbols(df, force=force)
    
    if not feature_data:
        logger.error("没有计算出任何特征")
        return
    
    logger.info(f"\n成功计算 {len(feature_data)} 个特征")
    
    # 打印汇总信息
    if print_summary:
        summary_file = Path(output_dir) / 'feature_summary.txt' if output_dir else None
        print_feature_summary(feature_data, output_file=summary_file)
    
    # 绘制图表
    logger.info("\n开始绘制图表...")
    plot_feature_trends(feature_data, output_dir=output_dir, figsize=figsize, dpi=dpi)
    
    logger.info("")
    for line in log_section("完成！所有特征图表已生成", width=80):
        logger.info(line)


def main():
    """主函数，支持命令行参数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='绘制所有 symbols 的特征走势图')
    parser.add_argument('--symbols', type=str, nargs='+', help='股票代码列表（可选，不指定则使用所有股票）')
    parser.add_argument('--output', type=str, help='输出目录（可选，不指定则显示图表）')
    parser.add_argument('--figsize', type=float, nargs=2, default=[14, 7], metavar=('WIDTH', 'HEIGHT'),
                       help='图表大小（默认: 14 7）')
    parser.add_argument('--dpi', type=int, default=150, help='图表分辨率（默认: 150）')
    parser.add_argument('--no-summary', action='store_true', help='不打印特征汇总信息')
    parser.add_argument('--force', action='store_true', help='强制重新计算，忽略缓存')
    
    args = parser.parse_args()
    
    output_dir = Path(args.output) if args.output else None
    
    plot_all_features(
        symbols=args.symbols,
        output_dir=output_dir,
        figsize=tuple(args.figsize),
        dpi=args.dpi,
        print_summary=not args.no_summary,
        force=args.force
    )


if __name__ == "__main__":
    main()

