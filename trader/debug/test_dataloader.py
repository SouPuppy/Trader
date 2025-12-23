"""
测试和可视化 Dataloader
比较所有 Dataloader 的输出
"""
import sys
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from typing import Optional, List

# 添加项目根目录到 Python 路径
project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from trader.dataloader import (
    dataloader_raw,
    dataloader_nullcomplete,
    dataloader_autocomplete,
    dataloader_ffill,
    dataloader_linear
)
from trader.logger import get_logger
import logging
from logging.handlers import RotatingFileHandler

# 设置日志级别为 DEBUG，让控制台输出 debug 信息
trader_logger = logging.getLogger('trader')
trader_logger.setLevel(logging.DEBUG)

# 设置控制台 handler 为 DEBUG（输出到控制台）
# 设置文件 handler 为 INFO（不记录到文件）
for handler in trader_logger.handlers:
    if isinstance(handler, logging.StreamHandler):
        # 控制台输出 DEBUG
        handler.setLevel(logging.DEBUG)
    elif isinstance(handler, RotatingFileHandler):
        # 文件只记录 INFO 及以上
        handler.setLevel(logging.INFO)

logger = get_logger(__name__)


def compare_dataloaders(
    symbol: str,
    start_date: str,
    end_date: str,
    feature_names: Optional[List[str]] = None,
    output_dir: Optional[Path] = None
):
    """
    比较所有 Dataloader 的输出并生成可视化图表
    
    Args:
        symbol: 股票代码
        start_date: 开始日期 (YYYY-MM-DD)
        end_date: 结束日期 (YYYY-MM-DD)
        feature_names: 要比较的特征名称列表，如果为 None 则比较所有特征
        output_dir: 输出目录，如果为 None 则使用默认目录
    """
    logger.info(f"开始比较 Dataloader: {symbol} from {start_date} to {end_date}")
    
    # 创建输出目录
    if output_dir is None:
        output_dir = project_root / 'output' / 'dataloader_comparison'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 加载所有 Dataloader 的数据
    logger.info("加载 dataloader_raw...")
    raw_loader = dataloader_raw(symbol)
    raw_df = raw_loader.load(start_date, end_date, feature_names)
    
    logger.info("加载 dataloader_nullcomplete...")
    nullcomplete_loader = dataloader_nullcomplete(symbol)
    nullcomplete_df = nullcomplete_loader.load(start_date, end_date, feature_names)
    
    logger.info("加载 dataloader_autocomplete...")
    autocomplete_loader = dataloader_autocomplete(symbol)
    autocomplete_df = autocomplete_loader.load(start_date, end_date, feature_names)
    
    logger.info("加载 dataloader_ffill...")
    ffill_loader = dataloader_ffill(symbol)
    ffill_df = ffill_loader.load(start_date, end_date, feature_names)
    
    logger.info("加载 dataloader_linear...")
    linear_loader = dataloader_linear(symbol)
    linear_df = linear_loader.load(start_date, end_date, feature_names)
    
    # 获取要比较的特征
    if feature_names is None:
        feature_names = raw_df.columns.tolist()
    
    # 为每个特征生成对比图
    for feature_name in feature_names:
        if feature_name not in raw_df.columns:
            logger.warning(f"特征 {feature_name} 不存在，跳过")
            continue
        
        logger.info(f"生成特征 {feature_name} 的对比图...")
        
        # 创建图表 - 5个子图
        fig, axes = plt.subplots(5, 1, figsize=(15, 18))
        fig.suptitle(f'{symbol} - {feature_name} Comparison', fontsize=16, fontweight='bold')
        # 调整布局，给标题留出空间
        fig.subplots_adjust(top=0.96)
        
        # dataloader_raw
        ax1 = axes[0]
        raw_data = raw_df[feature_name]
        ax1.plot(raw_data.index, raw_data.values, 'o-', label='Raw', markersize=3, linewidth=1)
        ax1.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        ax1.set_title('dataloader_raw (Raw Data, Holidays as None)', fontsize=12)
        ax1.set_ylabel(feature_name, fontsize=10)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        ax1.xaxis.set_major_locator(mdates.WeekdayLocator(interval=2))
        plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        # dataloader_nullcomplete
        ax2 = axes[1]
        nullcomplete_data = nullcomplete_df[feature_name]
        ax2.plot(nullcomplete_data.index, nullcomplete_data.values, 's-', 
                label='NullComplete', markersize=3, linewidth=1, color='orange')
        # 标记节假日（原始数据中为 None 的日期）
        holiday_mask = raw_data.isna()
        if holiday_mask.any():
            holiday_dates = raw_data.index[holiday_mask]
            ax2.scatter(holiday_dates, nullcomplete_data.loc[holiday_dates], 
                       marker='x', s=100, color='red', label='Holiday (None)', zorder=5)
        ax2.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        ax2.set_title('dataloader_nullcomplete (Fill Trading Day Nulls, Holidays Remain None)', fontsize=12)
        ax2.set_ylabel(feature_name, fontsize=10)
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        ax2.xaxis.set_major_locator(mdates.WeekdayLocator(interval=2))
        plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        # dataloader_autocomplete
        ax3 = axes[2]
        autocomplete_data = autocomplete_df[feature_name]
        ax3.plot(autocomplete_data.index, autocomplete_data.values, '^-', 
                label='AutoComplete', markersize=3, linewidth=1, color='green')
        # 标记节假日（原始数据中为 None 的日期）
        if holiday_mask.any():
            holiday_dates = raw_data.index[holiday_mask]
            ax3.scatter(holiday_dates, autocomplete_data.loc[holiday_dates], 
                       marker='x', s=100, color='red', label='Holiday (Interpolated)', zorder=5)
        ax3.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        ax3.set_title('dataloader_autocomplete (Fill All Nulls Including Holidays)', fontsize=12)
        ax3.set_ylabel(feature_name, fontsize=10)
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        ax3.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        ax3.xaxis.set_major_locator(mdates.WeekdayLocator(interval=2))
        plt.setp(ax3.xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        # dataloader_ffill
        ax4 = axes[3]
        ffill_data = ffill_df[feature_name]
        ax4.plot(ffill_data.index, ffill_data.values, 'd-', 
                label='FFill', markersize=3, linewidth=1, color='purple')
        # 标记节假日（原始数据中为 None 的日期）
        if holiday_mask.any():
            holiday_dates = raw_data.index[holiday_mask]
            ax4.scatter(holiday_dates, ffill_data.loc[holiday_dates], 
                       marker='x', s=100, color='red', label='Holiday (FFilled)', zorder=5)
        ax4.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        ax4.set_title('dataloader_ffill (Forward Fill All Nulls Including Holidays)', fontsize=12)
        ax4.set_ylabel(feature_name, fontsize=10)
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        ax4.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        ax4.xaxis.set_major_locator(mdates.WeekdayLocator(interval=2))
        plt.setp(ax4.xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        # dataloader_linear
        ax5 = axes[4]
        linear_data = linear_df[feature_name]
        ax5.plot(linear_data.index, linear_data.values, 'v-', 
                label='Linear', markersize=3, linewidth=1, color='brown')
        # 标记节假日（原始数据中为 None 的日期）
        if holiday_mask.any():
            holiday_dates = raw_data.index[holiday_mask]
            # 只标记那些在 linear_data 中存在的日期
            valid_holiday_dates = [d for d in holiday_dates if d in linear_data.index]
            if valid_holiday_dates:
                holiday_values = [linear_data.loc[d] for d in valid_holiday_dates]
                ax5.scatter(valid_holiday_dates, holiday_values, 
                           marker='x', s=100, color='red', label='Holiday (Linear Interpolated)', zorder=5)
        ax5.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        ax5.set_title('dataloader_linear (Linear Interpolation All Nulls Including Holidays)', fontsize=12)
        ax5.set_xlabel('Date', fontsize=10)
        ax5.set_ylabel(feature_name, fontsize=10)
        ax5.legend()
        ax5.grid(True, alpha=0.3)
        ax5.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        ax5.xaxis.set_major_locator(mdates.WeekdayLocator(interval=2))
        plt.setp(ax5.xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        plt.tight_layout()
        
        # 保存图表
        output_file = output_dir / f'{symbol}_{feature_name}_comparison.png'
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        logger.info(f"保存图表: {output_file}")
        plt.close()
    
    logger.info("Dataloader 比较完成！")


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='测试和可视化 Dataloader')
    parser.add_argument('--symbol', type=str, default='AAPL.O', help='股票代码')
    parser.add_argument('--start-date', type=str, required=True, help='开始日期 (YYYY-MM-DD)')
    parser.add_argument('--end-date', type=str, required=True, help='结束日期 (YYYY-MM-DD)')
    parser.add_argument('--features', type=str, nargs='+', help='要比较的特征名称列表（可选）')
    parser.add_argument('--output', type=str, help='输出目录（可选）')
    
    args = parser.parse_args()
    
    output_dir = Path(args.output) if args.output else None
    
    compare_dataloaders(
        symbol=args.symbol,
        start_date=args.start_date,
        end_date=args.end_date,
        feature_names=args.features,
        output_dir=output_dir
    )


if __name__ == '__main__':
    main()
