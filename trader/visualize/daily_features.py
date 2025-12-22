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

# 添加项目根目录到 Python 路径
project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from trader.config import DB_PATH
from trader.logger import get_logger
from trader.features import get_feature_names, get_feature
from trader.features import features  # 导入特征定义以触发注册

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


def compute_features_for_all_symbols(df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """
    为所有 symbols 计算所有特征
    
    Args:
        df: 包含所有股票数据的 DataFrame
        
    Returns:
        字典，key 为特征名称，value 为包含所有 symbols 该特征值的 DataFrame
        DataFrame 的列为 symbols，行为日期
    """
    feature_names = get_feature_names()
    results = {}
    
    # 按 symbol 分组处理
    for symbol in df['stock_code'].unique():
        symbol_df = df[df['stock_code'] == symbol].copy()
        
        # 确保 datetime 是索引
        if 'datetime' in symbol_df.columns:
            symbol_df = symbol_df.set_index('datetime').sort_index()
        elif not isinstance(symbol_df.index, pd.DatetimeIndex):
            logger.warning(f"无法为 {symbol} 设置 datetime 索引")
            continue
        
        # 为每个特征计算值
        for feature_name in feature_names:
            feature_spec = get_feature(feature_name)
            if feature_spec is None:
                continue
            
            try:
                # 计算特征
                feature_series = feature_spec.compute(symbol_df)
                
                # 初始化结果 DataFrame（如果还没有）
                if feature_name not in results:
                    results[feature_name] = pd.DataFrame()
                
                # 将结果添加到 DataFrame
                if not feature_series.empty:
                    # 确保索引对齐
                    if results[feature_name].empty:
                        results[feature_name] = pd.DataFrame(index=feature_series.index)
                    results[feature_name][symbol] = feature_series
                    
            except Exception as e:
                logger.warning(f"计算特征 {feature_name} 对于 {symbol} 时出错: {e}")
                continue
    
    # 确保所有 DataFrame 的索引都是 datetime，并且对齐
    for feature_name in results:
        if not results[feature_name].empty:
            results[feature_name].index = pd.to_datetime(results[feature_name].index)
            results[feature_name] = results[feature_name].sort_index()
    
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
        
        logger.info(f"绘制特征 {feature_name} 的趋势图（{len(valid_symbols)} 个symbols）...")
        
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
        
        # 获取颜色方案
        colors = get_color_palette(len(valid_symbols))
        max_data_points = 0
        
        # 为每个 symbol 绘制一条线，使用不同的样式组合
        for idx, symbol in enumerate(valid_symbols):
            if df[symbol].notna().sum() == 0:
                continue
            
            # 只绘制有数据的部分
            valid_data = df[symbol].dropna()
            if len(valid_data) > 0:
                max_data_points = max(max_data_points, len(valid_data))
                
                # 根据数据点数量决定是否显示标记
                marker_style = markers[idx % len(markers)] if len(valid_data) <= 50 else None
                marker_size = 4 if len(valid_data) <= 30 else 2
                
                # 使用不同的线型和颜色
                line_style = linestyles[idx % len(linestyles)]
                line_width = 2.0 if idx < 5 else 1.5  # 前5个symbols用更粗的线
                
                ax.plot(
                    valid_data.index, 
                    valid_data.values, 
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
                f"  ✓ {feature_name:20s} | Symbols: {len(valid_symbols):3d} | "
                f"Data Points: {total_points:5d} | Status: OK"
            )
        else:
            summary_lines.append(
                f"  ✗ {feature_name:20s} | Symbols:    0 | "
                f"Data Points:     0 | Status: NO DATA"
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
    print_summary: bool = True
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
    logger.info("=" * 80)
    logger.info("开始特征可视化流程")
    logger.info("=" * 80)
    
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
    feature_data = compute_features_for_all_symbols(df)
    
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
    
    logger.info("\n" + "=" * 80)
    logger.info("完成！所有特征图表已生成")
    logger.info("=" * 80)


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
    
    args = parser.parse_args()
    
    output_dir = Path(args.output) if args.output else None
    
    plot_all_features(
        symbols=args.symbols,
        output_dir=output_dir,
        figsize=tuple(args.figsize),
        dpi=args.dpi,
        print_summary=not args.no_summary
    )


if __name__ == "__main__":
    main()

