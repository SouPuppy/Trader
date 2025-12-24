"""
Predictor 可视化脚本
显示所有股票的预测结果，包括实际价格和预测价格的对比
"""
import sys
from pathlib import Path
from typing import List, Optional, Dict, Tuple
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta

# 添加项目根目录到 Python 路径
project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from trader.predictor.Predictor import Predictor
from trader.backtest.market import Market
from trader.logger import get_logger
from trader.predictor.config_loader import get_train_stocks, get_test_stocks

logger = get_logger(__name__)


def get_trained_stocks() -> List[str]:
    """
    获取用于可视化的股票代码列表
    
    优先使用共享模型，如果共享模型存在，返回测试集股票（用于评估模型性能）
    否则返回单独模型的股票列表
    
    Returns:
        股票代码列表
    """
    from trader.config import PROJECT_ROOT
    weights_dir = PROJECT_ROOT / 'weights'
    
    if not weights_dir.exists():
        return []
    
    # 首先检查共享模型
    lstm_dir = weights_dir / 'LSTM'
    shared_model_path = lstm_dir / 'model.pth'
    shared_scaler_path = lstm_dir / 'scaler.pkl'
    
    if shared_model_path.exists() and shared_scaler_path.exists():
        # 使用共享模型，返回测试集股票（用于评估模型性能）
        try:
            from trader.predictor.config_loader import get_test_stocks
            stocks = get_test_stocks()
            if stocks:
                logger.info(f"使用共享模型，测试集股票列表: {', '.join(stocks)}")
                return sorted(stocks)
        except Exception as e:
            logger.warning(f"无法从配置文件获取测试集股票列表: {e}")
        
        # 如果无法从配置获取，使用默认的10支测试股票
        default_test_stocks = [
            'ASML.O', 'EBAY.O', 'ENPH.O', 'FAST.O', 'JD.O',
            'MRNA.O', 'NFLX.O', 'PDD.O', 'PYPL.O', 'TMUS.O'
        ]
        logger.info(f"使用默认测试集股票列表: {', '.join(default_test_stocks)}")
        return default_test_stocks
    
    # 如果没有共享模型，查找单独模型
    model_files = list(weights_dir.glob('lstm_*.pth'))
    stocks = []
    
    for model_file in model_files:
        # 从文件名提取股票代码: lstm_AAPL_O.pth -> AAPL.O
        stock_code = model_file.stem.replace('lstm_', '').replace('_', '.')
        stocks.append(stock_code)
    
    return sorted(stocks)


def load_predictor(stock_code: str) -> Optional[Predictor]:
    """
    加载已训练的预测器
    
    优先使用共享模型，如果共享模型不存在，则使用单独模型
    
    Args:
        stock_code: 股票代码
    
    Returns:
        Predictor 实例，如果加载失败则返回 None
    """
    try:
        # 首先尝试加载共享模型
        from trader.config import PROJECT_ROOT
        weights_dir = PROJECT_ROOT / 'weights'
        lstm_dir = weights_dir / 'LSTM'
        shared_model_path = lstm_dir / 'model.pth'
        shared_scaler_path = lstm_dir / 'scaler.pkl'
        
        if shared_model_path.exists() and shared_scaler_path.exists():
            # 使用共享模型
            logger.info(f"加载共享模型用于股票: {stock_code}")
            predictor = Predictor(stock_code="shared", use_close_only=True)
            predictor.load_model()
            # 设置股票代码用于后续使用
            predictor.stock_code = stock_code
            return predictor
        else:
            # 使用单独模型
            logger.info(f"加载单独模型用于股票: {stock_code}")
            predictor = Predictor(stock_code=stock_code, use_close_only=True)
            predictor.load_model()
            return predictor
    except Exception as e:
        logger.error(f"加载股票 {stock_code} 的模型失败: {e}")
        return None


def prepare_features_for_prediction(data: pd.DataFrame) -> pd.DataFrame:
    """
    为预测准备特征数据
    
    Args:
        data: 原始价格数据
    
    Returns:
        包含所有必需特征的数据框
    """
    # 确保数据按日期排序
    if 'datetime' in data.columns:
        data = data.sort_values('datetime')
        data = data.set_index('datetime')
    elif not isinstance(data.index, pd.DatetimeIndex):
        data = data.sort_index()
    
    # 计算收益率特征
    if 'prev_close' in data.columns:
        data['ret_1d'] = data['close_price'] / data['prev_close'] - 1
    else:
        data['ret_1d'] = data['close_price'].pct_change()
    
    data['ret_5d'] = data['close_price'].pct_change(5)
    data['ret_20d'] = data['close_price'].pct_change(20)
    
    # 计算日内特征
    if 'prev_close' in data.columns:
        data['range_pct'] = (data['high_price'] - data['low_price']) / data['prev_close']
        data['gap_pct'] = (data['open_price'] - data['prev_close']) / data['prev_close']
    else:
        data['range_pct'] = (data['high_price'] - data['low_price']) / data['close_price'].shift(1)
        data['gap_pct'] = (data['open_price'] - data['close_price'].shift(1)) / data['close_price'].shift(1)
    
    data['close_to_open'] = data['close_price'] / data['open_price'] - 1
    
    # 计算波动率特征
    data['vol_20d'] = data['ret_1d'].rolling(window=20).std()
    data['vol_60d'] = data['ret_1d'].rolling(window=60).std()
    
    if 'volume' in data.columns:
        volume_mean_20d = data['volume'].rolling(window=20).mean()
        volume_std_20d = data['volume'].rolling(window=20).std()
        data['vol_z_20d'] = (data['volume'] - volume_mean_20d) / (volume_std_20d + 1e-8)
    else:
        data['vol_z_20d'] = 0
    
    # 填充缺失值
    data = data.ffill().fillna(0)
    
    return data


def generate_predictions(
    stock_code: str,
    predictor: Predictor,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    lookback_days: int = 100
) -> Tuple[pd.DataFrame, List[float], List[str]]:
    """
    生成预测结果
    
    Args:
        stock_code: 股票代码
        predictor: 预测器实例
        start_date: 开始日期
        end_date: 结束日期
        lookback_days: 用于预测的历史数据天数
    
    Returns:
        (历史数据, 预测值列表, 预测日期列表)
    """
    market = Market()
    
    # 获取历史数据
    if end_date:
        # 获取到 end_date 的数据
        data = market.get_price_data(stock_code, None, end_date)
    else:
        # 获取最新数据
        data = market.get_price_data(stock_code)
    
    if data.empty:
        logger.warning(f"股票 {stock_code} 没有数据")
        return pd.DataFrame(), [], []
    
    # 确保数据按日期排序
    if 'datetime' in data.columns:
        data = data.sort_values('datetime')
        if not isinstance(data.index, pd.DatetimeIndex):
            data = data.set_index('datetime')
    elif not isinstance(data.index, pd.DatetimeIndex):
        data = data.sort_index()
    
    # 确保有 close_price 列
    if 'close_price' not in data.columns:
        logger.warning(f"股票 {stock_code} 没有 close_price 列")
        return data, [], []
    
    # 确保有足够的数据
    if len(data) < predictor.seq_len + 1:
        logger.warning(f"股票 {stock_code} 数据不足，需要至少 {predictor.seq_len + 1} 天")
        return data, [], []
    
    # 获取 close_price 数组
    close_prices = data['close_price'].values
    
    # 生成预测
    predictions = []
    prediction_dates = []
    
    # 检查是否使用收益率模式
    use_returns = getattr(predictor, 'use_returns', False)
    
    # 从 seq_len 天之后开始预测（如果使用收益率模式，需要 seq_len+1 天）
    start_idx = predictor.seq_len + 1 if use_returns else predictor.seq_len
    for i in range(start_idx, min(len(data), start_idx + lookback_days)):
        try:
            # 获取前 seq_len 天的 close_price 用于预测
            # 如果使用收益率模式，需要 seq_len+1 个价格点
            window_size = predictor.seq_len + 1 if use_returns else predictor.seq_len
            window_close_prices = close_prices[i - window_size:i]
            
            # 预测下一天的价格
            if predictor.use_close_only:
                # 使用 close_price 数组
                prediction = predictor.predict(close_prices=window_close_prices)
            else:
                # 使用多特征（需要准备特征数据）
                window_data = data.iloc[i - predictor.seq_len:i]
                data_with_features = prepare_features_for_prediction(window_data.copy())
                prediction = predictor.predict(data=data_with_features)
            
            predictions.append(prediction)
            
            # 获取预测日期（下一天，即 i 对应的日期）
            if i < len(data):
                pred_date = data.index[i]
                if isinstance(pred_date, pd.Timestamp):
                    prediction_dates.append(pred_date.strftime('%Y-%m-%d'))
                else:
                    prediction_dates.append(str(pred_date))
        except Exception as e:
            logger.warning(f"预测股票 {stock_code} 第 {i} 天时出错: {e}")
            continue
    
    return data, predictions, prediction_dates


def plot_predictions(
    stock_code: str,
    data: pd.DataFrame,
    predictions: List[float],
    prediction_dates: List[str],
    output_dir: Optional[Path] = None,
    figsize: tuple = (14, 8),
    dpi: int = 150
):
    """
    绘制预测结果图表
    
    Args:
        stock_code: 股票代码
        data: 历史数据
        predictions: 预测值列表
        prediction_dates: 预测日期列表
        output_dir: 输出目录
        figsize: 图表大小
        dpi: 图表分辨率
    """
    if not predictions or not prediction_dates:
        logger.warning(f"股票 {stock_code} 没有预测数据")
        return
    
    # 设置中文字体
    plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize, dpi=dpi, height_ratios=[3, 1])
    
    # 准备数据
    pred_dates = pd.to_datetime(prediction_dates)
    actual_prices = []
    
    # 获取实际价格（预测日期对应的实际收盘价）
    for date_str in prediction_dates:
        date = pd.to_datetime(date_str)
        if date in data.index:
            actual_prices.append(data.loc[date, 'close_price'])
        else:
            # 如果找不到精确日期，找最近的
            closest_idx = data.index.get_indexer([date], method='nearest')[0]
            if closest_idx >= 0:
                actual_prices.append(data.iloc[closest_idx]['close_price'])
            else:
                actual_prices.append(np.nan)
    
    # 计算预测误差
    errors = [pred - actual for pred, actual in zip(predictions, actual_prices) if not np.isnan(actual)]
    error_pcts = [(pred - actual) / actual * 100 for pred, actual in zip(predictions, actual_prices) if not np.isnan(actual)]
    
    # 绘制价格对比
    ax1.plot(data.index, data['close_price'], label='实际价格', color='blue', linewidth=2, alpha=0.7)
    ax1.plot(pred_dates, predictions, label='预测价格', color='red', linewidth=2, marker='o', markersize=4, alpha=0.8)
    ax1.plot(pred_dates, actual_prices, label='实际价格（预测日）', color='green', linewidth=1.5, marker='s', markersize=3, alpha=0.6)
    
    ax1.set_title(f'{stock_code} - Predictor 预测结果', fontsize=16, fontweight='bold', pad=15)
    ax1.set_xlabel('日期', fontsize=12)
    ax1.set_ylabel('价格', fontsize=12)
    ax1.legend(loc='best', fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # 格式化 x 轴日期
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    ax1.xaxis.set_major_locator(mdates.WeekdayLocator(interval=7))
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    # 绘制预测误差
    if errors:
        error_dates = [d for d, a in zip(pred_dates, actual_prices) if not np.isnan(a)]
        ax2.bar(error_dates, error_pcts, alpha=0.6, color='orange', width=timedelta(days=1))
        ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
        ax2.set_xlabel('日期', fontsize=12)
        ax2.set_ylabel('预测误差 (%)', fontsize=12)
        ax2.grid(True, alpha=0.3)
        ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        ax2.xaxis.set_major_locator(mdates.WeekdayLocator(interval=7))
        plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    # 添加统计信息
    if errors:
        mae = np.mean(np.abs(errors))
        mape = np.mean(np.abs(error_pcts))
        rmse = np.sqrt(np.mean([e**2 for e in errors]))
        
        # 使用 r'...' 原始字符串，并转义 $ 符号，或者使用 usetex=False
        stats_text = f'MAE: \\${mae:.2f} | MAPE: {mape:.2f}% | RMSE: \\${rmse:.2f}'
        ax1.text(0.02, 0.98, stats_text, transform=ax1.transAxes,
                fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7),
                usetex=False)  # 禁用 LaTeX 渲染，避免 $ 符号解析问题
    
    plt.tight_layout()
    
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        output_file = output_dir / f'{stock_code.replace(".", "_")}_predictions.png'
        plt.savefig(output_file, dpi=dpi, bbox_inches='tight', facecolor='white')
        logger.info(f"保存图表到: {output_file}")
        plt.close()
    else:
        plt.show()


def visualize_all_stocks(
    stocks: Optional[List[str]] = None,
    output_dir: Optional[Path] = None,
    lookback_days: int = 100,
    figsize: tuple = (14, 8),
    dpi: int = 150
):
    """
    可视化所有股票的预测结果
    
    Args:
        stocks: 股票代码列表，如果为 None 则使用所有已训练的股票
        output_dir: 输出目录
        lookback_days: 用于预测的历史数据天数
        figsize: 图表大小
        dpi: 图表分辨率
    """
    if stocks is None:
        stocks = get_trained_stocks()
    
    if not stocks:
        logger.error("没有找到已训练的模型")
        return
    
    logger.info(f"开始可视化 {len(stocks)} 支股票的预测结果")
    logger.info(f"股票列表: {', '.join(stocks)}")
    
    # 统计信息
    all_stats = []
    
    for i, stock_code in enumerate(stocks, 1):
        logger.info(f"\n[{i}/{len(stocks)}] 处理股票: {stock_code}")
        logger.info("-" * 80)
        
        # 加载预测器
        predictor = load_predictor(stock_code)
        if predictor is None:
            logger.warning(f"跳过股票 {stock_code}：无法加载模型")
            continue
        
        # 生成预测
        data, predictions, prediction_dates = generate_predictions(
            stock_code, predictor, lookback_days=lookback_days
        )
        
        if not predictions:
            logger.warning(f"跳过股票 {stock_code}：没有生成预测")
            continue
        
        # 绘制图表
        plot_predictions(
            stock_code, data, predictions, prediction_dates,
            output_dir=output_dir, figsize=figsize, dpi=dpi
        )
        
        # 计算统计信息
        if data is not None and not data.empty and predictions:
            # 获取实际价格
            actual_prices = []
            for date_str in prediction_dates:
                date = pd.to_datetime(date_str)
                if date in data.index:
                    actual_prices.append(data.loc[date, 'close_price'])
                else:
                    closest_idx = data.index.get_indexer([date], method='nearest')[0]
                    if closest_idx >= 0:
                        actual_prices.append(data.iloc[closest_idx]['close_price'])
                    else:
                        actual_prices.append(np.nan)
            
            # 计算准确度
            valid_pairs = [(p, a) for p, a in zip(predictions, actual_prices) if not np.isnan(a)]
            if valid_pairs:
                # 计算方向准确度（预测涨跌方向是否正确）
                direction_correct = 0
                # 计算相对准确度（1 - 平均相对误差）
                relative_errors = []
                
                for i in range(1, len(valid_pairs)):
                    pred_prev, actual_prev = valid_pairs[i-1]
                    pred_curr, actual_curr = valid_pairs[i]
                    
                    # 方向准确度：预测的变化方向是否与实际一致
                    pred_direction = 1 if pred_curr > pred_prev else -1
                    actual_direction = 1 if actual_curr > actual_prev else -1
                    if pred_direction == actual_direction:
                        direction_correct += 1
                    
                    # 相对准确度：1 - |pred - actual| / actual
                    if actual_curr > 0:
                        relative_error = abs(pred_curr - actual_curr) / actual_curr
                        relative_errors.append(relative_error)
                
                direction_accuracy = direction_correct / (len(valid_pairs) - 1) * 100 if len(valid_pairs) > 1 else 0
                avg_relative_accuracy = (1 - np.mean(relative_errors)) * 100 if relative_errors else 0
                
                stats = {
                    'stock_code': stock_code,
                    'predictions': len(valid_pairs),
                    'direction_accuracy': direction_accuracy,
                    'relative_accuracy': avg_relative_accuracy
                }
                all_stats.append(stats)
    
    # 分别统计训练集和测试集的准确度
    if all_stats:
        try:
            train_stocks = set(get_train_stocks())
            test_stocks = set(get_test_stocks())
        except Exception as e:
            logger.warning(f"无法获取训练集/测试集列表: {e}")
            train_stocks = set()
            test_stocks = set()
        
        train_stats = [s for s in all_stats if s['stock_code'] in train_stocks]
        test_stats = [s for s in all_stats if s['stock_code'] in test_stocks]
        
        logger.info("\n" + "=" * 80)
        logger.info("准确度统计")
        logger.info("=" * 80)
        
        # 训练集统计
        if train_stats:
            logger.info(f"\n训练集统计 ({len(train_stats)} 支股票):")
            logger.info(f"{'股票代码':<12} {'预测数':<8} {'train_direction_accuracy (%)':<25} {'train_relative_accuracy (%)':<25}")
            logger.info("-" * 80)
            for stats in train_stats:
                logger.info(
                    f"{stats['stock_code']:<12} {stats['predictions']:<8} "
                    f"{stats['direction_accuracy']:<25.2f} {stats['relative_accuracy']:<25.2f}"
                )
            avg_train_direction = np.mean([s['direction_accuracy'] for s in train_stats])
            avg_train_relative = np.mean([s['relative_accuracy'] for s in train_stats])
            logger.info("-" * 80)
            logger.info(f"{'平均':<12} {'':<8} {avg_train_direction:<25.2f} {avg_train_relative:<25.2f}")
        
        # 测试集统计
        if test_stats:
            logger.info(f"\n测试集统计 ({len(test_stats)} 支股票):")
            logger.info(f"{'股票代码':<12} {'预测数':<8} {'test_direction_accuracy (%)':<25} {'test_relative_accuracy (%)':<25}")
            logger.info("-" * 80)
            for stats in test_stats:
                logger.info(
                    f"{stats['stock_code']:<12} {stats['predictions']:<8} "
                    f"{stats['direction_accuracy']:<25.2f} {stats['relative_accuracy']:<25.2f}"
                )
            avg_test_direction = np.mean([s['direction_accuracy'] for s in test_stats])
            avg_test_relative = np.mean([s['relative_accuracy'] for s in test_stats])
            logger.info("-" * 80)
            logger.info(f"{'平均':<12} {'':<8} {avg_test_direction:<25.2f} {avg_test_relative:<25.2f}")
        
        logger.info("=" * 80)
    
    logger.info(f"\n可视化完成！共处理 {len(stocks)} 支股票")


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='可视化 Predictor 预测结果')
    parser.add_argument(
        '--stocks',
        type=str,
        nargs='+',
        default=None,
        help='股票代码列表（可选，不指定则使用所有已训练的股票）'
    )
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='输出目录（可选，不指定则显示图表）'
    )
    parser.add_argument(
        '--lookback',
        type=int,
        default=100,
        help='用于预测的历史数据天数（默认: 100）'
    )
    parser.add_argument(
        '--figsize',
        type=float,
        nargs=2,
        default=[14, 8],
        metavar=('WIDTH', 'HEIGHT'),
        help='图表大小（默认: 14 8）'
    )
    parser.add_argument(
        '--dpi',
        type=int,
        default=150,
        help='图表分辨率（默认: 150）'
    )
    
    args = parser.parse_args()
    
    output_dir = Path(args.output) if args.output else None
    
    visualize_all_stocks(
        stocks=args.stocks,
        output_dir=output_dir,
        lookback_days=args.lookback,
        figsize=tuple(args.figsize),
        dpi=args.dpi
    )


if __name__ == '__main__':
    main()

