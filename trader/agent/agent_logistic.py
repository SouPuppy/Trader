"""
Logistic Regression 逻辑回归策略 Agent
使用逻辑回归模型预测股票未来收益，作为 baseline
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Optional
from datetime import datetime, timedelta
import sqlite3
from pathlib import Path
import sys

# 添加项目根目录到 Python 路径
project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

try:
    from tqdm import tqdm
except ImportError:
    # 如果 tqdm 未安装，使用一个简单的替代实现
    def tqdm(iterable, desc=None, total=None, **kwargs):
        if desc:
            print(f"{desc}...")
        return iterable

try:
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.impute import SimpleImputer
except ImportError:
    raise ImportError(
        "scikit-learn is required for LogisticAgent. "
        "Please install it with: pip install scikit-learn"
    )

from trader.agent.TradingAgent import TradingAgent
from trader.backtest.engine import BacktestEngine
from trader.config import DB_PATH
from trader.logger import get_logger
from trader.features.registry import get_feature_names

logger = get_logger(__name__)


class LogisticAgent(TradingAgent):
    """
    逻辑回归策略 Agent
    使用逻辑回归模型预测股票未来收益是否为正
    """
    
    def __init__(
        self,
        name: str = "LogisticAgent",
        feature_names: Optional[List[str]] = None,
        train_window_days: int = 252,  # 训练窗口：约1年交易日
        prediction_horizon: int = 5,  # 预测未来5天的收益
        ret_threshold: float = 0.0,  # 收益阈值：未来收益 > threshold 为正样本
        retrain_frequency: int = 20,  # 每20个交易日重新训练一次
        max_position_weight: float = 0.1,
        min_score_threshold: float = 0.0,
        max_total_weight: float = 1.0,
        train_test_split_ratio: float = 0.7  # 训练/测试分割比例（默认70%用于训练）
    ):
        """
        初始化 Logistic Agent
        
        Args:
            name: Agent 名称
            feature_names: 使用的特征名称列表，如果为 None 则使用默认特征
            train_window_days: 训练窗口大小（交易日数）
            prediction_horizon: 预测未来多少天的收益
            ret_threshold: 收益阈值，未来收益 > threshold 为正样本
            retrain_frequency: 重新训练频率（每N个交易日）
            max_position_weight: 单个股票最大配置比例
            min_score_threshold: 最小 score 阈值
            max_total_weight: 总配置比例上限
            train_test_split_ratio: 训练/测试分割比例（默认0.7，即70%用于训练，30%用于测试）
        """
        super().__init__(
            name=name,
            max_position_weight=max_position_weight,
            min_score_threshold=min_score_threshold,
            max_total_weight=max_total_weight
        )
        
        # 默认特征列表（技术指标）
        if feature_names is None:
            self.feature_names = [
                'ret_1d', 'ret_5d', 'ret_20d',
                'vol_20d', 'vol_60d', 'vol_z_20d',
                'range_pct', 'gap_pct', 'close_to_open',
                'pe_ratio_ttm', 'pb_ratio', 'ps_ratio_ttm'
            ]
        else:
            self.feature_names = feature_names
        
        self.train_window_days = train_window_days
        self.prediction_horizon = prediction_horizon
        self.ret_threshold = ret_threshold
        self.retrain_frequency = retrain_frequency
        
        # 模型相关
        self.model: Optional[LogisticRegression] = None
        self.scaler: Optional[StandardScaler] = None
        self.imputer: Optional[SimpleImputer] = None
        self.is_trained = False
        
        # 训练状态
        self.last_train_date: Optional[str] = None
        self.train_count = 0
        self.trading_day_count = 0
        self._data_insufficient_warned: set = set()  # 记录已警告的数据不足情况，避免重复警告
        
        # 训练/测试分割比例（用于准备训练数据时的分割）
        # 注意：实际的训练/测试分割日期由 BacktestEngine 统一管理
        self.train_test_split_ratio = train_test_split_ratio
    
    def _load_historical_data(
        self,
        stock_code: str,
        end_date: str,
        lookback_days: int
    ) -> pd.DataFrame:
        """
        加载历史数据（用于训练）
        
        Args:
            stock_code: 股票代码
            end_date: 结束日期（格式: YYYY-MM-DD）
            lookback_days: 回看天数（最小需要的天数）
            
        Returns:
            DataFrame: 包含日期和价格的数据，从最早可用日期到 end_date
        """
        try:
            conn = sqlite3.connect(DB_PATH)
            
            # 查询历史数据：加载从最早日期到 end_date 的所有数据
            # 不使用 LIMIT，而是加载所有可用数据，以便有足够的数据进行训练
            query = """
                SELECT datetime, close_price
                FROM raw_data
                WHERE stock_code = ? AND datetime <= ?
                ORDER BY datetime ASC
            """
            df = pd.read_sql_query(query, conn, params=(stock_code, end_date))
            
            # 查询数据库中的最早和最晚日期，用于诊断
            query_range = """
                SELECT MIN(datetime) as min_date, MAX(datetime) as max_date, COUNT(*) as count
                FROM raw_data
                WHERE stock_code = ?
            """
            range_info = pd.read_sql_query(query_range, conn, params=(stock_code,))
            conn.close()
            
            if df.empty:
                if not range_info.empty and range_info.iloc[0]['count'] > 0:
                    min_date = range_info.iloc[0]['min_date']
                    max_date = range_info.iloc[0]['max_date']
                    logger.warning(
                        f"数据库中 {stock_code} 的数据范围: {min_date} 至 {max_date}, "
                        f"但查询 end_date={end_date} 时未找到数据"
                    )
                return pd.DataFrame()
            
            df['datetime'] = pd.to_datetime(df['datetime'])
            df = df.sort_values('datetime').reset_index(drop=True)
            
            # 记录数据范围，用于诊断
            if len(df) < lookback_days:
                min_date = df.iloc[0]['datetime'].strftime('%Y-%m-%d')
                max_date = df.iloc[-1]['datetime'].strftime('%Y-%m-%d')
                logger.debug(
                    f"加载历史数据: {stock_code}, "
                    f"日期范围: {min_date} 至 {max_date}, "
                    f"共 {len(df)} 天, "
                    f"需要至少 {lookback_days} 天"
                )
            
            return df
        except Exception as e:
            logger.error(f"加载历史数据时出错: {e}", exc_info=True)
            return pd.DataFrame()
    
    def _get_future_return(
        self,
        price_data: pd.DataFrame,
        current_idx: int
    ) -> Optional[float]:
        """
        计算未来收益
        
        Args:
            price_data: 价格数据 DataFrame
            current_idx: 当前索引
            
        Returns:
            未来N天的收益率，如果数据不足则返回 None
        """
        if current_idx + self.prediction_horizon >= len(price_data):
            return None
        
        current_price = price_data.iloc[current_idx]['close_price']
        future_price = price_data.iloc[current_idx + self.prediction_horizon]['close_price']
        
        if pd.isna(current_price) or pd.isna(future_price) or current_price == 0:
            return None
        
        return (future_price / current_price) - 1.0
    
    def _prepare_training_data(
        self,
        stock_code: str,
        end_date: str,
        engine: BacktestEngine
    ) -> Optional[tuple]:
        """
        准备训练数据（只使用前70%的数据）
        
        Args:
            stock_code: 股票代码
            end_date: 结束日期
            engine: 回测引擎（用于获取特征）
            
        Returns:
            (X, y) 元组，X 是特征矩阵，y 是标签数组，如果失败则返回 None
        """
        # 加载历史价格数据（从最早日期到 end_date 的所有数据）
        price_data = self._load_historical_data(
            stock_code, end_date, self.train_window_days
        )
        
        # 检查是否有足够的数据进行训练
        # 至少需要 prediction_horizon + 一些样本（比如至少50个样本）才能训练
        min_samples_needed = self.prediction_horizon + 50  # 至少需要50个训练样本
        if len(price_data) < min_samples_needed:
            # 只在第一次尝试训练时记录警告，避免重复警告
            warning_key = f"{stock_code}_{end_date}"
            if warning_key not in self._data_insufficient_warned:
                logger.debug(
                    f"数据不足，无法训练: {stock_code}, "
                    f"需要至少 {min_samples_needed} 天（用于构建至少50个样本），"
                    f"实际 {len(price_data)} 天。"
                    f"请确保数据库中有足够的历史数据（至少到 {end_date} 之前 {min_samples_needed} 天）"
                )
                self._data_insufficient_warned.add(warning_key)
            return None
        
        # 计算训练/测试分割点（70%/30%）
        # 使用所有可用的历史数据，而不是固定的 train_window_days
        total_samples = len(price_data) - self.prediction_horizon
        train_samples = int(total_samples * self.train_test_split_ratio)
        
        # 确保训练样本数至少为 min_samples_needed - prediction_horizon
        min_train_samples = max(50, min_samples_needed - self.prediction_horizon)
        if train_samples < min_train_samples:
            # 只在第一次尝试训练时记录警告，避免重复警告
            warning_key = f"{stock_code}_{end_date}_samples"
            if warning_key not in self._data_insufficient_warned:
                logger.debug(
                    f"训练样本数不足: {stock_code}, "
                    f"需要至少 {min_train_samples} 个训练样本，"
                    f"实际 {train_samples} 个。"
                    f"数据总量: {len(price_data)} 天, "
                    f"训练集需要至少 {min_train_samples + self.prediction_horizon} 天"
                )
                self._data_insufficient_warned.add(warning_key)
            return None
        
        # 注意：训练/测试分割日期由 BacktestEngine 统一管理，这里不再设置
        
        X_list = []
        y_list = []
        dates_list = []
        
        # 只使用前70%的数据进行训练（使用 tqdm 显示进度）
        for i in tqdm(
            range(train_samples),
            desc=f"准备训练数据 ({stock_code}, 前70%)",
            total=train_samples,
            leave=False
        ):
            date_str = price_data.iloc[i]['datetime'].strftime('%Y-%m-%d')
            
            # 获取特征值
            features = []
            missing_features = False
            
            for feature_name in self.feature_names:
                try:
                    # 直接使用 date 参数获取历史特征，不需要修改 engine.current_date
                    feature_value = engine.get_feature(feature_name, stock_code, date=date_str)
                    if feature_value is None:
                        missing_features = True
                        break
                    features.append(feature_value)
                except Exception as e:
                    logger.debug(f"获取特征 {feature_name} 失败: {e}")
                    missing_features = True
                    break
            
            if missing_features:
                continue
            
            # 计算未来收益作为标签
            future_ret = self._get_future_return(price_data, i)
            if future_ret is None:
                continue
            
            # 标签：未来收益 > threshold 为正样本（1），否则为负样本（0）
            label = 1 if future_ret > self.ret_threshold else 0
            
            X_list.append(features)
            y_list.append(label)
            dates_list.append(date_str)
        
        if len(X_list) == 0:
            logger.warning(f"无法构建训练样本: {stock_code}")
            return None
        
        # 检查实际构建的样本数是否满足最小要求
        min_train_samples = max(50, min_samples_needed - self.prediction_horizon)
        if len(X_list) < min_train_samples:
            # 只在第一次尝试训练时记录警告，避免重复警告
            warning_key = f"{stock_code}_{end_date}_built"
            if warning_key not in self._data_insufficient_warned:
                logger.debug(
                    f"实际构建的训练样本数不足: {stock_code}, "
                    f"需要至少 {min_train_samples} 个训练样本，"
                    f"实际构建了 {len(X_list)} 个（理论计算 {train_samples} 个）。"
                    f"可能是由于特征缺失导致部分样本被跳过"
                )
                self._data_insufficient_warned.add(warning_key)
            return None
        
        X = np.array(X_list)
        y = np.array(y_list)
        
        logger.info(
            f"准备训练数据完成: {stock_code}, "
            f"样本数: {len(X)}, 正样本: {np.sum(y)}, 负样本: {len(y) - np.sum(y)}"
        )
        
        return (X, y)
    
    def _train_model(self, stock_code: str, end_date: str, engine: BacktestEngine):
        """
        训练逻辑回归模型
        
        Args:
            stock_code: 股票代码
            end_date: 训练数据结束日期
            engine: 回测引擎
        """
        logger.info(f"开始训练逻辑回归模型: {stock_code}, 截止日期: {end_date}")
        
        # 准备训练数据
        training_data = self._prepare_training_data(stock_code, end_date, engine)
        if training_data is None:
            # 数据不足的情况已经在 _prepare_training_data 中记录，这里不需要重复记录
            return
        
        X, y = training_data
        
        logger.info(f"数据预处理中... (样本数: {len(X)}, 特征数: {len(self.feature_names)})")
        
        # 数据预处理
        # 1. 填充缺失值
        self.imputer = SimpleImputer(strategy='mean')
        X_imputed = self.imputer.fit_transform(X)
        
        # 2. 标准化特征
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X_imputed)
        
        # 3. 训练模型
        logger.info("训练逻辑回归模型...")
        self.model = LogisticRegression(
            max_iter=1000,
            random_state=42,
            class_weight='balanced'  # 处理类别不平衡
        )
        self.model.fit(X_scaled, y)
        
        # 评估模型
        train_score = self.model.score(X_scaled, y)
        logger.info(
            f"模型训练完成: {stock_code}, "
            f"训练准确率: {train_score:.4f}, "
            f"特征数: {len(self.feature_names)}, "
            f"正样本率: {np.mean(y):.2%}"
        )
        
        self.is_trained = True
        self.last_train_date = end_date
        self.train_count += 1
    
    def _has_sufficient_data(self, stock_code: str, end_date: str) -> bool:
        """
        快速检查是否有足够的数据进行训练
        
        Args:
            stock_code: 股票代码
            end_date: 结束日期
            
        Returns:
            是否有足够的数据
        """
        min_samples_needed = self.prediction_horizon + 50
        price_data = self._load_historical_data(stock_code, end_date, min_samples_needed)
        return len(price_data) >= min_samples_needed
    
    def _should_retrain(self, current_date: str, stock_code: str = None) -> bool:
        """
        判断是否需要重新训练
        
        Args:
            current_date: 当前日期
            stock_code: 股票代码（可选，用于检查数据是否足够）
            
        Returns:
            是否需要重新训练
        """
        # 如果从未训练过，需要先检查数据是否足够
        if not self.is_trained or self.model is None:
            # 如果提供了股票代码，先检查数据是否足够
            if stock_code:
                if not self._has_sufficient_data(stock_code, current_date):
                    return False  # 数据不足，不尝试训练
            return True
        
        # 如果达到重新训练频率，需要训练
        if self.trading_day_count % self.retrain_frequency == 0:
            # 如果提供了股票代码，先检查数据是否足够
            if stock_code:
                if not self._has_sufficient_data(stock_code, current_date):
                    return False  # 数据不足，不尝试训练
            return True
        
        return False
    
    def score(self, stock_code: str, engine: BacktestEngine) -> float:
        """
        计算股票的 score（使用逻辑回归模型预测）
        
        Args:
            stock_code: 股票代码
            engine: 回测引擎
            
        Returns:
            float: 分数，范围 [-1, 1]
                  - 正数表示预测未来收益为正的概率较高
                  - 负数表示预测未来收益为负的概率较高
        """
        # 判断当前是否在测试期（从 engine 获取）
        is_in_test_period = engine.is_in_test_period()
        
        # 检查是否需要重新训练（只在训练期重新训练）
        if not is_in_test_period and self._should_retrain(engine.current_date, stock_code):
            self._train_model(stock_code, engine.current_date, engine)
        
        # 如果模型未训练成功，返回 0
        if not self.is_trained or self.model is None:
            return 0.0
        
        # 获取当前特征
        features = []
        for feature_name in self.feature_names:
            try:
                feature_value = engine.get_feature(feature_name, stock_code)
                if feature_value is None:
                    # 特征缺失，返回 0
                    logger.debug(f"特征 {feature_name} 缺失: {stock_code}")
                    return 0.0
                features.append(feature_value)
            except Exception as e:
                logger.debug(f"获取特征 {feature_name} 失败: {e}")
                return 0.0
        
        # 预处理特征
        try:
            X = np.array([features])
            X_imputed = self.imputer.transform(X)
            X_scaled = self.scaler.transform(X_imputed)
            
            # 预测概率
            proba = self.model.predict_proba(X_scaled)[0]
            
            # 返回正样本的概率，映射到 [-1, 1] 范围
            # proba[1] 是正样本（未来收益 > threshold）的概率
            positive_prob = proba[1] if len(proba) > 1 else 0.5
            
            # 将概率 [0, 1] 映射到 score [-1, 1]
            # 概率 > 0.5 为正，概率 < 0.5 为负
            score = (positive_prob - 0.5) * 2.0
            
            return score
        except Exception as e:
            logger.error(f"预测时出错: {e}", exc_info=True)
            return 0.0
    
    def on_date(self, engine: BacktestEngine, date: str):
        """
        每个交易日的回调函数
        
        Args:
            engine: 回测引擎
            date: 当前日期
        """
        self.trading_day_count += 1

