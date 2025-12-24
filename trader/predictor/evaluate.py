"""
预测模型评估模块
包含三层评估：基线对比 + 信息含量 + 策略结果
"""
import sys
from pathlib import Path
from typing import List, Optional, Dict, Tuple
import pandas as pd
import numpy as np
from scipy.stats import spearmanr
from datetime import datetime

# 添加项目根目录到 Python 路径
project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from trader.predictor.Predictor import Predictor
from trader.backtest.market import Market
from trader.dataloader import dataloader_linear
from trader.logger import get_logger

logger = get_logger(__name__)


def compute_baseline_predictions(
    data: pd.DataFrame,
    method: str = 'naive_price'
) -> np.ndarray:
    """
    计算基线预测
    
    Args:
        data: 包含 close_price 的 DataFrame，索引为日期
        method: 基线方法
            - 'naive_price': 预测明天价=今天价
            - 'naive_return': 预测明天收益=0（或过去20日均值）
    
    Returns:
        预测的对数收益率数组
    """
    close_prices = data['close_price'].values
    
    if method == 'naive_price':
        # 预测明天价=今天价，即 log_return = 0
        # 需要 len(data) - 1 个预测（因为第一个价格没有前一个价格）
        predictions = np.zeros(len(close_prices) - 1)
    
    elif method == 'naive_return':
        # 预测明天收益=过去20日均值
        log_returns = np.diff(np.log(close_prices + 1e-8))
        predictions = []
        
        for i in range(len(log_returns)):
            # 使用过去20天的平均对数收益率（如果有的话）
            lookback = min(20, i)
            if lookback > 0:
                mean_return = np.mean(log_returns[i - lookback:i])
            else:
                mean_return = 0.0
            predictions.append(mean_return)
        
        predictions = np.array(predictions)
    
    else:
        raise ValueError(f"未知的基线方法: {method}")
    
    return predictions


def compute_direction_accuracy(
    predicted_returns: np.ndarray,
    actual_returns: np.ndarray
) -> float:
    """
    计算方向准确率（针对 sign(y)）
    
    Args:
        predicted_returns: 预测的对数收益率数组
        actual_returns: 实际的对数收益率数组
    
    Returns:
        方向准确率（0-1之间）
    """
    if len(predicted_returns) != len(actual_returns):
        raise ValueError("预测和实际数组长度不匹配")
    
    if len(predicted_returns) == 0:
        return 0.0
    
    # 计算方向：1 表示上涨，-1 表示下跌，0 表示不变
    pred_directions = np.sign(predicted_returns)
    actual_directions = np.sign(actual_returns)
    
    # 计算方向一致的样本数
    correct = np.sum(pred_directions == actual_directions)
    
    return correct / len(predicted_returns)


def compute_ic(
    predicted_returns: np.ndarray,
    actual_returns: np.ndarray
) -> float:
    """
    计算 IC（Information Coefficient）：corr(预测收益, 真实收益)
    
    Args:
        predicted_returns: 预测的对数收益率数组
        actual_returns: 实际的对数收益率数组
    
    Returns:
        IC 值（Pearson 相关系数）
    """
    if len(predicted_returns) != len(actual_returns):
        raise ValueError("预测和实际数组长度不匹配")
    
    if len(predicted_returns) < 2:
        return 0.0
    
    # 计算 Pearson 相关系数
    correlation = np.corrcoef(predicted_returns, actual_returns)[0, 1]
    
    # 处理 NaN
    if np.isnan(correlation):
        return 0.0
    
    return correlation


def compute_rank_ic(
    predicted_returns: np.ndarray,
    actual_returns: np.ndarray
) -> float:
    """
    计算 Rank IC：Spearman corr（对排序更稳定）
    
    Args:
        predicted_returns: 预测的对数收益率数组
        actual_returns: 实际的对数收益率数组
    
    Returns:
        Rank IC 值（Spearman 相关系数）
    """
    if len(predicted_returns) != len(actual_returns):
        raise ValueError("预测和实际数组长度不匹配")
    
    if len(predicted_returns) < 2:
        return 0.0
    
    # 计算 Spearman 相关系数
    correlation, _ = spearmanr(predicted_returns, actual_returns)
    
    # 处理 NaN
    if np.isnan(correlation):
        return 0.0
    
    return correlation


def compute_simple_strategy_returns(
    predicted_returns: np.ndarray,
    actual_returns: np.ndarray,
    transaction_cost: float = 0.001
) -> Dict[str, float]:
    """
    计算最简策略收益
    
    策略规则：预测收益 > 0 就持有，≤0 就空仓
    
    注意：predicted_returns 和 actual_returns 都是对数收益率（log return）
    
    Args:
        predicted_returns: 预测的对数收益率数组
        actual_returns: 实际的对数收益率数组
        transaction_cost: 交易手续费率（默认 0.1%，这是简单收益率成本）
    
    Returns:
        包含策略收益指标的字典
    """
    if len(predicted_returns) != len(actual_returns):
        raise ValueError("预测和实际数组长度不匹配")
    
    if len(predicted_returns) == 0:
        return {
            'total_return': 0.0,
            'annualized_return': 0.0,
            'max_drawdown': 0.0,
            'sharpe_ratio': 0.0,
            'win_rate': 0.0
        }
    
    # 将对数收益率转换为简单收益率（用于复利计算）
    # log return -> simple return: simple = exp(log) - 1
    simple_actual_returns = np.exp(actual_returns) - 1.0
    
    # 策略信号：预测收益 > 0 就持有（1），否则空仓（0）
    signals = (predicted_returns > 0).astype(float)
    
    # 计算策略收益（考虑手续费）
    # 手续费是简单收益率成本，需要从简单收益率中扣除
    strategy_returns = []
    position = 0.0  # 当前持仓状态：0=空仓，1=持有
    
    for i in range(len(signals)):
        signal = signals[i]
        r = simple_actual_returns[i]
        
        # 如果信号改变，需要交易（产生手续费）
        traded = (signal != position)
        cost = transaction_cost if traded else 0.0
        position = signal
        
        # 如果持有，获得实际收益并扣除手续费；如果空仓，只损失手续费
        strategy_return = position * r - cost
        strategy_returns.append(strategy_return)
    
    strategy_returns = np.array(strategy_returns)
    
    # 计算累计收益（使用简单收益率复利）
    cumulative_returns = np.cumprod(1 + strategy_returns) - 1
    total_return = cumulative_returns[-1]
    
    # 计算年化收益（假设252个交易日）
    n_days = len(strategy_returns)
    if n_days > 0:
        annualized_return = (1 + total_return) ** (252 / n_days) - 1
    else:
        annualized_return = 0.0
    
    # 计算最大回撤
    running_max = np.maximum.accumulate(1 + cumulative_returns)
    drawdown = (1 + cumulative_returns) / running_max - 1
    max_drawdown = np.min(drawdown)
    
    # 计算夏普比率（假设无风险利率为0）
    # 使用简单收益率的均值和标准差
    if len(strategy_returns) > 1:
        mean_return = np.mean(strategy_returns)
        std_return = np.std(strategy_returns)
        if std_return > 0:
            sharpe_ratio = np.sqrt(252) * mean_return / std_return
        else:
            sharpe_ratio = 0.0
    else:
        sharpe_ratio = 0.0
    
    # 计算胜率
    positive_returns = strategy_returns[strategy_returns > 0]
    win_rate = len(positive_returns) / len(strategy_returns) if len(strategy_returns) > 0 else 0.0
    
    return {
        'total_return': total_return,
        'annualized_return': annualized_return,
        'max_drawdown': max_drawdown,
        'sharpe_ratio': sharpe_ratio,
        'win_rate': win_rate,
        'strategy_returns': strategy_returns,
        'cumulative_returns': cumulative_returns
    }


def evaluate_predictor(
    stock_code: str,
    predictor: Predictor,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    lookback_days: int = 100,
    transaction_cost: float = 0.001
) -> Dict[str, any]:
    """
    评估预测器性能（三层评估）
    
    注意：使用与训练时相同的数据加载器（dataloader_linear）以确保数据一致性
    
    Args:
        stock_code: 股票代码
        predictor: 预测器实例
        start_date: 开始日期
        end_date: 结束日期
        lookback_days: 用于评估的历史数据天数
        transaction_cost: 交易手续费率
    
    Returns:
        包含所有评估指标的字典
    """
    # 使用与训练时相同的数据加载器（dataloader_linear）以确保数据一致性
    # 这会补齐节假日并使用线性插值，与训练时的数据处理方式一致
    loader = dataloader_linear(stock_code)
    
    # 确定日期范围
    if end_date is None:
        # 如果没有指定结束日期，使用当前日期
        end_date = datetime.now().strftime('%Y-%m-%d')
    
    if start_date is None:
        # 如果没有指定开始日期，需要足够的历史数据
        # 至少需要 seq_len + lookback_days + 1 天的数据
        # 这里我们使用一个足够早的日期
        from datetime import timedelta
        end_date_obj = pd.to_datetime(end_date)
        start_date_obj = end_date_obj - timedelta(days=365 * 2)  # 提前2年以确保有足够数据
        start_date = start_date_obj.strftime('%Y-%m-%d')
    
    # 加载数据（使用线性插值补齐节假日）
    data = loader.load(start_date, end_date, feature_names=['close_price'], force=False)
    
    if data.empty:
        logger.warning(f"股票 {stock_code} 没有数据")
        return {}
    
    # 确保数据按日期排序
    if not isinstance(data.index, pd.DatetimeIndex):
        data = data.sort_index()
    else:
        data = data.sort_index()
    
    # 确保有 close_price 列
    if 'close_price' not in data.columns:
        logger.warning(f"股票 {stock_code} 没有 close_price 列")
        return {}
    
    # 确保有足够的数据
    if len(data) < predictor.seq_len + 2:
        logger.warning(f"股票 {stock_code} 数据不足，需要至少 {predictor.seq_len + 2} 天")
        return {}
    
    # 获取 close_price 数组
    close_prices = data['close_price'].values
    
    # 生成预测和对齐的实际收益率
    # 注意：模型用 close[i-seq_len:i] 预测 close[i]
    # 所以预测的对数收益率应该是 log(predicted_close[i] / actual_close[i-1])
    # 实际的对数收益率应该是 log(actual_close[i] / actual_close[i-1])
    predicted_log_returns = []
    actual_log_returns_aligned = []
    prediction_dates = []
    
    use_log_returns = getattr(predictor, 'use_log_returns', False)
    start_idx = predictor.seq_len  # 从 seq_len 开始（因为需要 seq_len 个历史数据）
    
    # 确保有足够的数据
    if len(close_prices) < start_idx + 1:
        logger.warning(f"股票 {stock_code} 数据不足，需要至少 {start_idx + 1} 天，当前只有 {len(close_prices)} 天")
        return {}
    
    for i in range(start_idx, min(len(data), start_idx + lookback_days)):
        try:
            # 根据是否使用对数收益率模式决定窗口大小
            # 当 use_log_returns=True 时，predict 方法需要 seq_len+1 个价格点来计算 seq_len 个对数收益率
            # 当 use_log_returns=False 时，predict 方法需要 seq_len 个价格点
            if use_log_returns:
                # 对数收益率模式：需要 seq_len+1 个价格点
                window_start = i - predictor.seq_len
                window_end = i + 1  # 包含 i，共 seq_len+1 个点
                expected_window_size = predictor.seq_len + 1
            else:
                # 绝对价格模式：需要 seq_len 个价格点
                window_start = i - predictor.seq_len
                window_end = i  # 不包含 i，共 seq_len 个点
                expected_window_size = predictor.seq_len
            
            # 检查窗口起始索引是否有效
            if window_start < 0:
                logger.warning(
                    f"股票 {stock_code} 第 {i} 天：窗口起始索引无效，"
                    f"需要从 {window_start} 开始，但数据从 0 开始"
                )
                continue
            
            # 检查窗口结束索引是否有效
            if window_end > len(close_prices):
                logger.warning(
                    f"股票 {stock_code} 第 {i} 天：窗口结束索引超出范围，"
                    f"需要到 {window_end}，但数据只有 {len(close_prices)} 天"
                )
                continue
            
            # 获取窗口数据
            window_close_prices = close_prices[window_start:window_end]
            
            # 验证窗口大小
            if len(window_close_prices) != expected_window_size:
                logger.warning(
                    f"股票 {stock_code} 第 {i} 天：窗口数据大小不匹配，"
                    f"需要 {expected_window_size} 天，当前只有 {len(window_close_prices)} 天"
                )
                continue
            
            # 预测 close[i]（模型用窗口数据预测 close[i]）
            if predictor.use_close_only:
                prediction_price = predictor.predict(close_prices=window_close_prices)
            else:
                # 使用多特征（需要准备特征数据）
                window_data = data.iloc[window_start:i]
                from trader.predictor.visualize import prepare_features_for_prediction
                data_with_features = prepare_features_for_prediction(window_data.copy())
                prediction_price = predictor.predict(data=data_with_features)
            
            # 计算预测的对数收益率
            # 模型预测的是 close[i]，所以预测的对数收益率是 log(predicted_close[i] / actual_close[i-1])
            # 实际的对数收益率是 log(actual_close[i] / actual_close[i-1])
            prev_price = close_prices[i - 1]
            actual_price = close_prices[i]
            
            if prev_price > 0 and actual_price > 0 and prediction_price > 0:
                predicted_log_return = np.log(prediction_price / prev_price)
                actual_log_return = np.log(actual_price / prev_price)
            else:
                predicted_log_return = 0.0
                actual_log_return = 0.0
            
            predicted_log_returns.append(predicted_log_return)
            actual_log_returns_aligned.append(actual_log_return)
            
            # 获取预测日期（对应的是 close[i] 的日期）
            if i < len(data):
                pred_date = data.index[i]
                if isinstance(pred_date, pd.Timestamp):
                    prediction_dates.append(pred_date.strftime('%Y-%m-%d'))
                else:
                    prediction_dates.append(str(pred_date))
        except Exception as e:
            logger.warning(f"预测股票 {stock_code} 第 {i} 天时出错: {e}")
            continue
    
    if len(predicted_log_returns) == 0:
        logger.warning(f"股票 {stock_code} 没有生成预测")
        return {}
    
    predicted_log_returns = np.array(predicted_log_returns)
    actual_log_returns_aligned = np.array(actual_log_returns_aligned)
    
    if len(actual_log_returns_aligned) != len(predicted_log_returns):
        # 如果长度不匹配，截取较短的长度
        min_len = min(len(actual_log_returns_aligned), len(predicted_log_returns))
        actual_log_returns_aligned = actual_log_returns_aligned[:min_len]
        predicted_log_returns = predicted_log_returns[:min_len]
    
    predicted_log_returns = np.array(predicted_log_returns)
    actual_log_returns_aligned = np.array(actual_log_returns_aligned)
    
    # A. 基线对比
    # 计算基线预测（需要包含足够的历史数据）
    # baseline 预测的是 log return，需要与 predicted_log_returns 对齐
    # 对于每一天 i（从 start_idx 开始），baseline 预测的是 log(close[i] / close[i-1])
    baseline_naive_price_aligned = []
    baseline_naive_return_aligned = []
    
    # 计算历史 log returns（用于 naive_return baseline）
    historical_log_returns = np.diff(np.log(close_prices + 1e-8))
    
    for i in range(start_idx, min(len(data), start_idx + len(predicted_log_returns))):
        # naive_price: 预测 log return = 0（即预测价格不变）
        baseline_naive_price_aligned.append(0.0)
        
        # naive_return: 预测 log return = 过去20天的平均 log return
        lookback_start = max(0, i - 20)
        if lookback_start < i:
            mean_return = np.mean(historical_log_returns[lookback_start:i])
        else:
            mean_return = 0.0
        baseline_naive_return_aligned.append(mean_return)
    
    baseline_naive_price_aligned = np.array(baseline_naive_price_aligned)
    baseline_naive_return_aligned = np.array(baseline_naive_return_aligned)
    
    # 确保长度一致
    min_len = min(len(predicted_log_returns), len(baseline_naive_price_aligned), len(baseline_naive_return_aligned))
    predicted_log_returns = predicted_log_returns[:min_len]
    actual_log_returns_aligned = actual_log_returns_aligned[:min_len]
    baseline_naive_price_aligned = baseline_naive_price_aligned[:min_len]
    baseline_naive_return_aligned = baseline_naive_return_aligned[:min_len]
    
    # B. 信息含量指标
    direction_accuracy = compute_direction_accuracy(predicted_log_returns, actual_log_returns_aligned)
    ic = compute_ic(predicted_log_returns, actual_log_returns_aligned)
    rank_ic = compute_rank_ic(predicted_log_returns, actual_log_returns_aligned)
    
    # 基线方向准确率
    baseline_naive_price_direction = compute_direction_accuracy(baseline_naive_price_aligned, actual_log_returns_aligned)
    baseline_naive_return_direction = compute_direction_accuracy(baseline_naive_return_aligned, actual_log_returns_aligned)
    
    # C. 策略结果
    strategy_metrics = compute_simple_strategy_returns(
        predicted_log_returns,
        actual_log_returns_aligned,
        transaction_cost
    )
    
    # 基线策略结果
    baseline_naive_price_strategy = compute_simple_strategy_returns(
        baseline_naive_price_aligned,
        actual_log_returns_aligned,
        transaction_cost
    )
    baseline_naive_return_strategy = compute_simple_strategy_returns(
        baseline_naive_return_aligned,
        actual_log_returns_aligned,
        transaction_cost
    )
    
    return {
        'stock_code': stock_code,
        'n_predictions': len(predicted_log_returns),
        # 基线对比
        'baseline_naive_price': {
            'direction_accuracy': baseline_naive_price_direction,
            'annualized_return': baseline_naive_price_strategy['annualized_return'],
            'sharpe_ratio': baseline_naive_price_strategy['sharpe_ratio'],
            'max_drawdown': baseline_naive_price_strategy['max_drawdown']
        },
        'baseline_naive_return': {
            'direction_accuracy': baseline_naive_return_direction,
            'annualized_return': baseline_naive_return_strategy['annualized_return'],
            'sharpe_ratio': baseline_naive_return_strategy['sharpe_ratio'],
            'max_drawdown': baseline_naive_return_strategy['max_drawdown']
        },
        # 信息含量
        'direction_accuracy': direction_accuracy,
        'ic': ic,
        'rank_ic': rank_ic,
        # 策略结果
        'strategy': {
            'annualized_return': strategy_metrics['annualized_return'],
            'sharpe_ratio': strategy_metrics['sharpe_ratio'],
            'max_drawdown': strategy_metrics['max_drawdown'],
            'win_rate': strategy_metrics['win_rate'],
            'total_return': strategy_metrics['total_return']
        },
        # 原始数据（用于进一步分析）
        'predicted_returns': predicted_log_returns,
        'actual_returns': actual_log_returns_aligned,
        'prediction_dates': prediction_dates
    }

