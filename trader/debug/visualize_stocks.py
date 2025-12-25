"""
可视化所有股票的价格走势图
将20支股票垂直排列在一个很长的图中
"""
import sys
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import sqlite3

# 添加项目根目录到 Python 路径
project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from trader.config import DB_PATH
from trader.logger import get_logger

logger = get_logger(__name__)


def load_all_stock_data(stock_code: str) -> pd.DataFrame:
    """
    从数据库加载股票的所有历史数据
    
    Args:
        stock_code: 股票代码
        
    Returns:
        DataFrame，包含股票数据，按日期排序
    """
    if not DB_PATH.exists():
        logger.error(f"数据库文件不存在: {DB_PATH}")
        return pd.DataFrame()
    
    try:
        conn = sqlite3.connect(DB_PATH)
        
        query = """
            SELECT 
                datetime,
                stock_code,
                close_price,
                volume
            FROM raw_data 
            WHERE stock_code = ?
            ORDER BY datetime ASC
        """
        df = pd.read_sql_query(query, conn, params=(stock_code,))
        
        conn.close()
        
        if df.empty:
            logger.warning(f"未找到股票数据: {stock_code}")
            return pd.DataFrame()
        
        # 确保 datetime 是 datetime 类型
        if 'datetime' in df.columns:
            df['datetime'] = pd.to_datetime(df['datetime'])
            df = df.sort_values('datetime')
        
        return df
        
    except Exception as e:
        logger.error(f"加载股票数据时出错: {e}", exc_info=True)
        return pd.DataFrame()


def visualize_all_stocks(
    stock_codes: list = None,
    output_path: Path = None
):
    """
    可视化所有股票的价格走势图，垂直排列
    
    Args:
        stock_codes: 股票代码列表
        output_path: 输出文件路径
    """
    if stock_codes is None:
        stock_codes = [
            "AAPL.O", "MSFT.O", "GOOGL.O", "AMZN.O", "NVDA.O",
            "TSLA.O", "META.O", "ASML.O", "MRNA.O", "NFLX.O",
            "AMD.O", "INTC.O", "ADBE.O", "CRM.N", "ORCL.N",
            "CSCO.O", "JPM.N", "V.N", "MA.N", "WMT.N"
        ]
    
    if output_path is None:
        output_path = project_root / 'output' / 'stock' / 'stocks.png'
    
    # 确保输出目录存在
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"开始加载 {len(stock_codes)} 支股票的数据...")
    
    # 加载所有股票数据
    stock_data_dict = {}
    for stock_code in stock_codes:
        logger.info(f"加载 {stock_code} 的数据...")
        df = load_all_stock_data(stock_code)
        if not df.empty:
            stock_data_dict[stock_code] = df
        else:
            logger.warning(f"股票 {stock_code} 没有数据，跳过")
    
    if not stock_data_dict:
        logger.error("没有找到任何股票数据")
        return
    
    logger.info(f"成功加载 {len(stock_data_dict)} 支股票的数据")
    
    # 设置中文字体
    plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    
    # 创建子图：20支股票垂直排列
    n_stocks = len(stock_data_dict)
    fig, axes = plt.subplots(n_stocks, 1, figsize=(16, n_stocks * 3), dpi=150)
    
    # 如果只有一支股票，axes 不是数组
    if n_stocks == 1:
        axes = [axes]
    
    # 为每支股票绘制价格走势
    for idx, (stock_code, df) in enumerate(sorted(stock_data_dict.items())):
        ax = axes[idx]
        
        # 绘制收盘价
        ax.plot(df['datetime'], df['close_price'], linewidth=1.5, color='blue', alpha=0.8)
        
        # 设置标题和标签
        ax.set_title(f'{stock_code}', fontsize=12, fontweight='bold', pad=10)
        ax.set_ylabel('价格', fontsize=10)
        ax.grid(True, alpha=0.3)
        
        # 格式化 x 轴日期
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right', fontsize=8)
        
        # 只在最后一个子图显示 x 轴标签
        if idx < n_stocks - 1:
            ax.set_xlabel('')
            ax.set_xticklabels([])
        else:
            ax.set_xlabel('日期', fontsize=10)
        
        # 添加统计信息
        if len(df) > 0:
            min_price = df['close_price'].min()
            max_price = df['close_price'].max()
            current_price = df['close_price'].iloc[-1]
            first_price = df['close_price'].iloc[0]
            total_return = (current_price - first_price) / first_price * 100 if first_price > 0 else 0
            
            stats_text = f'当前: ${current_price:.2f} | 最高: ${max_price:.2f} | 最低: ${min_price:.2f} | 总收益: {total_return:+.2f}%'
            ax.text(0.02, 0.95, stats_text, transform=ax.transAxes,
                   fontsize=9, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))
    
    # 设置整体标题
    fig.suptitle('20支股票价格走势图', fontsize=16, fontweight='bold', y=0.995)
    
    # 调整布局
    plt.tight_layout(rect=[0, 0, 1, 0.99])
    
    # 保存图片
    logger.info(f"保存图片到: {output_path}")
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    
    logger.info(f"图片已保存: {output_path}")


if __name__ == "__main__":
    visualize_all_stocks()

