"""
Predictor 预测器类
使用 LSTM 模型根据21天的数据预测下一天的价位
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, List, Dict, Tuple
from sklearn.preprocessing import StandardScaler
import pickle
import sqlite3
from datetime import datetime
from trader.config import PROJECT_ROOT, DB_PATH
from trader.logger import get_logger
from trader.backtest.market import Market
from trader.dataloader import dataloader_linear

logger = get_logger(__name__)


class LSTMPredictor(nn.Module):
    """
    LSTM 预测模型
    根据21天的特征数据预测下一天的收盘价
    """
    
    def __init__(
        self,
        input_size: int,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2,
        output_size: int = 1
    ):
        """
        初始化 LSTM 模型
        
        Args:
            input_size: 输入特征维度（特征数量）
            hidden_size: LSTM 隐藏层大小（默认64）
            num_layers: LSTM 层数（默认2）
            dropout: Dropout 比例（默认0.2）
            output_size: 输出维度（默认1，预测一个价位）
        """
        super(LSTMPredictor, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size
        
        # LSTM 层
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True  # 输入格式为 (batch, seq_len, features)
        )
        
        # 全连接层：将 LSTM 输出映射到预测值
        self.fc = nn.Linear(hidden_size, output_size)
        
        # Dropout 层
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: 输入张量，形状为 (batch_size, seq_len, input_size)
               其中 seq_len=21（21天的数据）
        
        Returns:
            预测值，形状为 (batch_size, output_size)
        """
        # LSTM 前向传播
        # lstm_out: (batch_size, seq_len, hidden_size)
        # hidden: (num_layers, batch_size, hidden_size)
        # cell: (num_layers, batch_size, hidden_size)
        lstm_out, (hidden, cell) = self.lstm(x)
        
        # 使用最后一个时间步的输出
        # lstm_out[:, -1, :] 形状为 (batch_size, hidden_size)
        last_output = lstm_out[:, -1, :]
        
        # 应用 dropout
        last_output = self.dropout(last_output)
        
        # 通过全连接层得到预测值
        output = self.fc(last_output)
        
        return output
    
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """
        预测方法（推理模式）
        
        Args:
            x: 输入张量，形状为 (batch_size, seq_len, input_size) 或 (seq_len, input_size)
        
        Returns:
            预测值
        """
        self.eval()
        with torch.no_grad():
            # 如果输入是二维的，添加 batch 维度
            if x.dim() == 2:
                x = x.unsqueeze(0)
            return self.forward(x)


class StockDataset(Dataset):
    """
    股票数据集类
    用于准备训练数据
    """
    
    def __init__(self, sequences: np.ndarray, targets: np.ndarray):
        """
        初始化数据集
        
        Args:
            sequences: 输入序列，形状为 (n_samples, seq_len, n_features)
            targets: 目标值，形状为 (n_samples,)
        """
        self.sequences = torch.FloatTensor(sequences)
        self.targets = torch.FloatTensor(targets)
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        return self.sequences[idx], self.targets[idx]


class ClosePriceDataLoader:
    """
    从 dataloader_linear 加载 close_price 数据的 DataLoader
    用于 LSTM 训练和测试
    使用线性插值处理完的数据，包括节假日的补全
    """
    
    def __init__(self, stock_code: str, seq_len: int = 21):
        """
        初始化 DataLoader
        
        Args:
            stock_code: 股票代码
            seq_len: 序列长度（默认21天）
        """
        self.stock_code = stock_code
        self.seq_len = seq_len
        # 使用 dataloader_linear 来加载处理完的数据
        self.linear_loader = dataloader_linear(stock_code)
    
    def load_close_data(
        self,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> pd.DataFrame:
        """
        从 dataloader_linear 加载 close_price 数据（已处理，包括线性插值补全）
        
        Args:
            start_date: 开始日期（格式: YYYY-MM-DD），如果为 None 则需要从数据库获取最早日期
            end_date: 结束日期（格式: YYYY-MM-DD），如果为 None 则需要从数据库获取最晚日期
        
        Returns:
            DataFrame，索引为日期，包含 close_price 列，按日期升序排列
            所有 null 值（包括节假日）都已使用线性插值补全
        """
        try:
            # 如果没有指定日期范围，需要从数据库获取可用日期范围
            if start_date is None or end_date is None:
                conn = sqlite3.connect(DB_PATH)
                query = """
                    SELECT MIN(datetime) as min_date, MAX(datetime) as max_date
                    FROM raw_data
                    WHERE stock_code = ?
                """
                cursor = conn.execute(query, (self.stock_code,))
                row = cursor.fetchone()
                conn.close()
                
                if not row or row[0] is None:
                    raise ValueError(f"未找到股票 {self.stock_code} 的数据")
                
                if start_date is None:
                    start_date = row[0]
                if end_date is None:
                    end_date = row[1]
            
            # 使用 dataloader_linear 加载数据（只加载 close_price 特征）
            df = self.linear_loader.load(
                start_date=start_date,
                end_date=end_date,
                feature_names=['close_price'],
                force=False
            )
            
            if df.empty:
                raise ValueError(f"未找到股票 {self.stock_code} 的数据")
            
            # 确保 close_price 列存在
            if 'close_price' not in df.columns:
                raise ValueError(f"数据中缺少 close_price 列，可用列: {df.columns.tolist()}")
            
            # 确保 close_price 是数值类型
            df['close_price'] = pd.to_numeric(df['close_price'], errors='coerce')
            
            # dataloader_linear 已经处理了缺失值（线性插值），但为了安全起见，检查一下
            if df['close_price'].isna().any():
                logger.warning(f"仍有缺失值，使用前向和后向填充")
                df['close_price'] = df['close_price'].ffill().bfill().fillna(0)
            
            if len(df) < self.seq_len + 1:
                raise ValueError(
                    f"数据不足，需要至少 {self.seq_len + 1} 天的数据，"
                    f"当前只有 {len(df)} 天"
                )
            
            logger.info(
                f"成功从 dataloader_linear 加载 {self.stock_code} 的数据: "
                f"{len(df)} 条记录, "
                f"日期范围: {df.index.min()} 到 {df.index.max()}, "
                f"缺失值已通过线性插值补全"
            )
            
            return df
            
        except Exception as e:
            logger.error(f"从 dataloader_linear 加载数据时出错: {e}", exc_info=True)
            raise
    
    def create_sequences(
        self,
        data: pd.DataFrame,
        train_test_split: Optional[float] = None,
        train_test_split_date: Optional[str] = None
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        创建训练序列和目标值
        
        Args:
            data: 包含 close_price 的 DataFrame，索引为日期
            train_test_split: 训练集比例（0-1之间），如果指定则按比例分割
            train_test_split_date: 训练/测试集分割日期（格式: YYYY-MM-DD），
                                  如果指定则按日期分割
        
        Returns:
            (X_train, y_train, X_test, y_test): 训练和测试序列及目标值
        """
        close_prices = data['close_price'].values
        
        # 创建序列
        sequences = []
        targets = []
        
        for i in range(self.seq_len, len(close_prices)):
            sequences.append(close_prices[i - self.seq_len:i])
            targets.append(close_prices[i])
        
        sequences = np.array(sequences)
        targets = np.array(targets)
        
        # 分割训练集和测试集
        if train_test_split_date:
            # 按日期分割
            split_date = pd.to_datetime(train_test_split_date)
            data_dates = data.index
            split_idx = None
            
            # 找到第一个大于等于分割日期的数据点
            for i, date in enumerate(data_dates):
                if date >= split_date:
                    # 序列从 seq_len 开始，所以需要调整索引
                    # 如果数据索引是 i，对应的序列索引是 i - seq_len
                    split_idx = max(0, i - self.seq_len)
                    break
            
            if split_idx is None:
                # 如果所有日期都小于分割日期，使用所有数据作为训练集
                split_idx = len(sequences)
                logger.warning(
                    f"所有数据日期都小于分割日期 {train_test_split_date}，"
                    f"将使用所有数据作为训练集"
                )
            elif split_idx <= 0:
                raise ValueError(
                    f"无法找到合适的分割点，"
                    f"分割日期: {train_test_split_date}, "
                    f"数据日期范围: {data_dates.min()} 到 {data_dates.max()}, "
                    f"计算出的 split_idx: {split_idx}"
                )
            
            logger.info(
                f"按日期分割 ({train_test_split_date}): "
                f"训练集 {split_idx} 个样本, "
                f"测试集 {len(sequences) - split_idx} 个样本"
            )
            
        elif train_test_split is not None:
            # 按比例分割
            split_idx = int(len(sequences) * train_test_split)
            logger.info(
                f"按比例分割 ({train_test_split}): "
                f"训练集 {split_idx} 个样本, "
                f"测试集 {len(sequences) - split_idx} 个样本"
            )
        else:
            # 默认使用 80% 作为训练集
            split_idx = int(len(sequences) * 0.8)
            logger.info(
                f"默认分割 (0.8): "
                f"训练集 {split_idx} 个样本, "
                f"测试集 {len(sequences) - split_idx} 个样本"
            )
        
        X_train = sequences[:split_idx]
        y_train = targets[:split_idx]
        X_test = sequences[split_idx:]
        y_test = targets[split_idx:]
        
        return X_train, y_train, X_test, y_test


class Predictor:
    """
    预测器类
    使用 LSTM 模型根据21天的数据预测下一天的价位
    """
    
    def __init__(
        self,
        stock_code: str,
        feature_names: Optional[List[str]] = None,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2,
        seq_len: int = 21,
        model_path: Optional[Path] = None,
        use_close_only: bool = False
    ):
        """
        初始化预测器
        
        Args:
            stock_code: 股票代码
            feature_names: 使用的特征名称列表，如果为 None 则使用默认特征
            hidden_size: LSTM 隐藏层大小
            num_layers: LSTM 层数
            dropout: Dropout 比例
            seq_len: 序列长度（默认21天）
            model_path: 模型保存路径，如果为 None 则使用默认路径
            use_close_only: 如果为 True，只使用 close_price 作为特征
        """
        self.stock_code = stock_code
        self.seq_len = seq_len
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.use_close_only = use_close_only
        
        # 设置特征名称
        if use_close_only:
            # 只使用 close_price
            self.feature_names = ['close_price']
        elif feature_names is None:
            # 默认使用价格相关特征
            self.feature_names = [
                'close_price', 'open_price', 'high_price', 'low_price',
                'volume', 'prev_close',
                'ret_1d', 'ret_5d', 'ret_20d',
                'range_pct', 'gap_pct', 'close_to_open',
                'vol_20d', 'vol_60d', 'vol_z_20d'
            ]
        else:
            self.feature_names = feature_names
        
        self.input_size = len(self.feature_names)
        
        # 初始化 DataLoader
        self.data_loader = ClosePriceDataLoader(stock_code, seq_len)
        
        # 初始化模型
        self.model = LSTMPredictor(
            input_size=self.input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout
        )
        
        # 数据标准化器
        self.scaler_X = StandardScaler()
        self.scaler_y = StandardScaler()
        
        # 模型路径
        if model_path is None:
            weights_dir = PROJECT_ROOT / 'weights' / 'LSTM'
            weights_dir.mkdir(parents=True, exist_ok=True)
            if stock_code == "shared":
                # 共享模型使用固定路径
                self.model_path = weights_dir / 'model.pth'
                self.scaler_path = weights_dir / 'scaler.pkl'
            else:
                # 单独模型
                self.model_path = weights_dir / f'lstm_{stock_code.replace(".", "_")}.pth'
                self.scaler_path = weights_dir / f'scaler_{stock_code.replace(".", "_")}.pkl'
        else:
            self.model_path = model_path
            self.scaler_path = model_path.parent / f'scaler_{model_path.stem}.pkl'
        
        self.is_trained = False
    
    def prepare_data(
        self,
        data: pd.DataFrame,
        target_column: str = 'close_price'
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        准备训练数据
        
        Args:
            data: 包含特征和目标的数据框，索引为日期
            target_column: 目标列名（默认 'close_price'）
        
        Returns:
            (sequences, targets): 序列数据和目标值
        """
        # 确保数据按日期排序
        if 'datetime' in data.columns:
            data = data.sort_values('datetime')
        elif data.index.name == 'datetime' or isinstance(data.index, pd.DatetimeIndex):
            data = data.sort_index()
        
        # 选择特征列
        available_features = [f for f in self.feature_names if f in data.columns]
        if len(available_features) < len(self.feature_names):
            missing = set(self.feature_names) - set(available_features)
            logger.warning(f"缺少特征: {missing}，将使用可用特征: {available_features}")
            self.feature_names = available_features
            self.input_size = len(available_features)
        
        # 提取特征和目标
        X = data[self.feature_names].values
        y = data[target_column].values
        
        # 处理缺失值：使用前向填充
        X_df = pd.DataFrame(X)
        X_df = X_df.ffill().fillna(0)
        X = X_df.values
        
        y_series = pd.Series(y)
        y_series = y_series.ffill().fillna(0)
        y = y_series.values
        
        # 创建序列
        sequences = []
        targets = []
        
        for i in range(self.seq_len, len(X)):
            sequences.append(X[i - self.seq_len:i])
            targets.append(y[i])
        
        sequences = np.array(sequences)
        targets = np.array(targets)
        
        return sequences, targets
    
    def train(
        self,
        data: Optional[pd.DataFrame] = None,
        target_column: str = 'close_price',
        epochs: int = 50,
        batch_size: int = 32,
        learning_rate: float = 0.001,
        validation_split: float = 0.2,
        train_test_split: Optional[float] = None,
        train_test_split_date: Optional[str] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        verbose: bool = True,
        use_wandb: bool = False,
        wandb_project: str = "lstm-stock-predictor",
        wandb_run_name: Optional[str] = None
    ):
        """
        训练模型
        
        Args:
            data: 训练数据框，包含特征和目标。如果为 None，则从数据库加载
            target_column: 目标列名
            epochs: 训练轮数
            batch_size: 批次大小
            learning_rate: 学习率
            validation_split: 验证集比例（用于从训练集中再分出验证集）
            train_test_split: 训练/测试集比例（0-1之间），如果指定则按比例分割
            train_test_split_date: 训练/测试集分割日期（格式: YYYY-MM-DD），
                                  如果指定则按日期分割
            start_date: 开始日期（格式: YYYY-MM-DD），如果 data 为 None 则从数据库加载时使用
            end_date: 结束日期（格式: YYYY-MM-DD），如果 data 为 None 则从数据库加载时使用
            verbose: 是否打印训练过程
            use_wandb: 是否使用 wandb 记录训练过程
            wandb_project: wandb 项目名称
            wandb_run_name: wandb 运行名称，如果为 None 则自动生成
        """
        logger.info(f"开始训练 LSTM 模型，股票: {self.stock_code}")
        
        # 如果 data 为 None，从数据库加载
        if data is None:
            if self.use_close_only:
                # 只加载 close_price
                data = self.data_loader.load_close_data(start_date, end_date)
            else:
                # 加载完整数据
                market = Market()
                data = market.get_price_data(self.stock_code, start_date, end_date)
                if data.empty:
                    raise ValueError(f"未找到股票 {self.stock_code} 的数据")
        
        # 如果只使用 close_price，直接从 close_price 创建序列
        if self.use_close_only:
            X_train, y_train, X_test, y_test = self.data_loader.create_sequences(
                data,
                train_test_split=train_test_split,
                train_test_split_date=train_test_split_date
            )
            
            # 保存测试集供后续验证使用
            self.test_sequences = X_test
            self.test_targets = y_test
            
            # 重塑数据以匹配 LSTM 输入格式 (n_samples, seq_len, n_features)
            # 对于 close_only，n_features=1
            X_train = X_train.reshape(-1, self.seq_len, 1)
            X_test = X_test.reshape(-1, self.seq_len, 1)
            
            # 数据标准化
            # 重塑用于标准化
            X_train_reshaped = X_train.reshape(-1, 1)
            X_test_reshaped = X_test.reshape(-1, 1)
            
            self.scaler_X = StandardScaler()
            X_train_scaled = self.scaler_X.fit_transform(X_train_reshaped)
            X_train_scaled = X_train_scaled.reshape(-1, self.seq_len, 1)
            
            X_test_scaled = self.scaler_X.transform(X_test_reshaped)
            X_test_scaled = X_test_scaled.reshape(-1, self.seq_len, 1)
            
            # 标准化目标值
            self.scaler_y = StandardScaler()
            y_train_scaled = self.scaler_y.fit_transform(y_train.reshape(-1, 1)).flatten()
            y_test_scaled = self.scaler_y.transform(y_test.reshape(-1, 1)).flatten()
            
            # 保存测试集的标准化版本
            self.test_sequences_scaled = X_test_scaled
            self.test_targets_scaled = y_test_scaled
            
            # 从训练集中再分出验证集
            val_split_idx = int(len(X_train_scaled) * (1 - validation_split))
            X_train_final = X_train_scaled[:val_split_idx]
            y_train_final = y_train_scaled[:val_split_idx]
            X_val = X_train_scaled[val_split_idx:]
            y_val = y_train_scaled[val_split_idx:]
            
        else:
            # 使用原有的多特征方法
            sequences, targets = self.prepare_data(data, target_column)
            
            if len(sequences) == 0:
                raise ValueError("数据不足，无法创建训练序列")
            
            # 数据标准化
            n_samples, seq_len, n_features = sequences.shape
            sequences_reshaped = sequences.reshape(-1, n_features)
            sequences_scaled = self.scaler_X.fit_transform(sequences_reshaped)
            sequences_scaled = sequences_scaled.reshape(n_samples, seq_len, n_features)
            
            targets_scaled = self.scaler_y.fit_transform(targets.reshape(-1, 1)).flatten()
            
            # 分割训练集和测试集
            if train_test_split_date:
                # 按日期分割
                split_date = pd.to_datetime(train_test_split_date)
                data_dates = data.index if isinstance(data.index, pd.DatetimeIndex) else pd.to_datetime(data['datetime'])
                split_idx = None
                
                for i, date in enumerate(data_dates):
                    if date >= split_date:
                        split_idx = i - self.seq_len
                        break
                
                if split_idx is None or split_idx <= 0:
                    split_idx = int(len(sequences_scaled) * 0.8)
            elif train_test_split is not None:
                split_idx = int(len(sequences_scaled) * train_test_split)
            else:
                split_idx = int(len(sequences_scaled) * 0.8)
            
            X_train_scaled = sequences_scaled[:split_idx]
            y_train_scaled = targets_scaled[:split_idx]
            X_test_scaled = sequences_scaled[split_idx:]
            y_test_scaled = targets_scaled[split_idx:]
            
            # 保存测试集
            self.test_sequences = X_test_scaled
            self.test_targets = self.scaler_y.inverse_transform(y_test_scaled.reshape(-1, 1)).flatten()
            self.test_sequences_scaled = X_test_scaled
            self.test_targets_scaled = y_test_scaled
            
            # 从训练集中再分出验证集
            val_split_idx = int(len(X_train_scaled) * (1 - validation_split))
            X_train_final = X_train_scaled[:val_split_idx]
            y_train_final = y_train_scaled[:val_split_idx]
            X_val = X_train_scaled[val_split_idx:]
            y_val = y_train_scaled[val_split_idx:]
        
        # 初始化 wandb
        if use_wandb:
            try:
                import wandb
                if wandb_run_name is None:
                    wandb_run_name = f"{self.stock_code}_lstm_{epochs}epochs"
                
                # 构建 wandb 配置
                wandb_config = {
                    "stock_code": self.stock_code,
                    "hidden_size": self.hidden_size,
                    "num_layers": self.num_layers,
                    "dropout": self.dropout,
                    "seq_len": self.seq_len,
                    "input_size": self.input_size,
                    "epochs": epochs,
                    "batch_size": batch_size,
                    "learning_rate": learning_rate,
                    "validation_split": validation_split,
                    "feature_names": self.feature_names,
                    "target_column": target_column
                }
                
                # 如果是共享模型，添加股票代码列表
                if self.stock_code == "shared" and hasattr(self, '_train_stock_codes'):
                    wandb_config["stock_codes"] = self._train_stock_codes
                    wandb_config["num_stocks"] = len(self._train_stock_codes)
                
                # 检查是否已登录 wandb，决定使用在线还是离线模式
                try:
                    api = wandb.Api()
                    # 尝试获取用户信息，如果失败则使用离线模式
                    try:
                        viewer = wandb.api.viewer()
                        if viewer:
                            wandb_mode = "online"
                            logger.info(f"检测到 wandb 登录，使用在线模式")
                        else:
                            wandb_mode = "offline"
                            logger.info(f"未检测到 wandb 登录，使用离线模式")
                    except Exception:
                        wandb_mode = "offline"
                        logger.info(f"无法验证 wandb 登录状态，使用离线模式")
                except Exception:
                    wandb_mode = "offline"
                    logger.info(f"无法初始化 wandb API，使用离线模式")
                
                run = wandb.init(
                    project=wandb_project,
                    name=wandb_run_name,
                    mode=wandb_mode,
                    config=wandb_config
                )
                
                if wandb_mode == "online":
                    logger.info(f"已初始化 wandb (在线模式)，项目: {wandb_project}, 运行: {wandb_run_name}")
                else:
                    logger.info(f"已初始化 wandb (离线模式)，项目: {wandb_project}, 运行: {wandb_run_name}")
                    logger.info(f"离线数据保存在: {PROJECT_ROOT / 'wandb'}")
                    logger.info(f"要同步到云端，请运行: wandb sync {PROJECT_ROOT / 'wandb' / 'offline-run-*'}")
            except ImportError:
                logger.warning("wandb 未安装，跳过 wandb 记录。安装命令: pip install wandb")
                use_wandb = False
        
        # 创建数据集和数据加载器
        train_dataset = StockDataset(X_train_final, y_train_final)
        val_dataset = StockDataset(X_val, y_val)
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        # 重新初始化模型（因为 input_size 可能已更新）
        self.model = LSTMPredictor(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            dropout=self.dropout
        )
        
        # 优化器和损失函数
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        criterion = nn.MSELoss()
        
        # 训练循环
        best_val_loss = float('inf')
        train_losses = []
        val_losses = []
        
        for epoch in range(epochs):
            # 训练阶段
            self.model.train()
            train_loss = 0.0
            for batch_X, batch_y in train_loader:
                optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = criterion(outputs.squeeze(), batch_y)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
            
            train_loss /= len(train_loader)
            train_losses.append(train_loss)
            
            # 验证阶段
            self.model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for batch_X, batch_y in val_loader:
                    outputs = self.model(batch_X)
                    loss = criterion(outputs.squeeze(), batch_y)
                    val_loss += loss.item()
            
            val_loss /= len(val_loader)
            val_losses.append(val_loss)
            
            # 记录到 wandb
            if use_wandb:
                try:
                    import wandb
                    wandb.log({
                        "epoch": epoch + 1,
                        "train_loss": train_loss,
                        "val_loss": val_loss,
                        "learning_rate": learning_rate
                    })
                except ImportError:
                    pass
            
            # 保存最佳模型
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                self.save_model()
                if use_wandb:
                    try:
                        import wandb
                        wandb.log({"best_val_loss": best_val_loss})
                    except ImportError:
                        pass
            
            if verbose and (epoch + 1) % 10 == 0:
                logger.info(
                    f"Epoch [{epoch+1}/{epochs}], "
                    f"Train Loss: {train_loss:.6f}, "
                    f"Val Loss: {val_loss:.6f}"
                )
        
        self.is_trained = True
        logger.info(f"训练完成，最佳验证损失: {best_val_loss:.6f}")
        
        # 在测试集上评估
        if hasattr(self, 'test_sequences_scaled') and len(self.test_sequences_scaled) > 0:
            test_loss, test_metrics = self.evaluate_on_test_set()
            logger.info(f"测试集评估 - 损失: {test_loss:.6f}")
            logger.info(f"测试集评估 - {test_metrics}")
            
            if use_wandb:
                try:
                    import wandb
                    wandb.log({
                        "test_loss": test_loss,
                        **{f"test_{k}": v for k, v in test_metrics.items()}
                    })
                except ImportError:
                    pass
        
        # 结束 wandb 运行
        if use_wandb:
            try:
                import wandb
                wandb.log({
                    "final_train_loss": train_losses[-1],
                    "final_val_loss": val_losses[-1],
                    "best_val_loss": best_val_loss
                })
                wandb.finish()
            except ImportError:
                pass
    
    def evaluate_on_test_set(self) -> Tuple[float, Dict[str, float]]:
        """
        在测试集上评估模型性能
        
        Returns:
            (test_loss, metrics_dict): 测试损失和评估指标字典
        """
        if not self.is_trained:
            raise ValueError("模型尚未训练，请先调用 train() 方法")
        
        if not hasattr(self, 'test_sequences_scaled') or len(self.test_sequences_scaled) == 0:
            raise ValueError("测试集不存在，请先训练模型")
        
        self.model.eval()
        criterion = nn.MSELoss()
        
        # 转换为张量
        X_test_tensor = torch.FloatTensor(self.test_sequences_scaled)
        y_test_tensor = torch.FloatTensor(self.test_targets_scaled)
        
        # 创建测试数据加载器
        test_dataset = StockDataset(self.test_sequences_scaled, self.test_targets_scaled)
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
        
        # 计算测试损失
        test_loss = 0.0
        predictions = []
        actuals = []
        
        with torch.no_grad():
            for batch_X, batch_y in test_loader:
                outputs = self.model(batch_X)
                loss = criterion(outputs.squeeze(), batch_y)
                test_loss += loss.item()
                
                # 收集预测值和实际值（反标准化后）
                pred_scaled = outputs.squeeze().numpy()
                actual_scaled = batch_y.numpy()
                
                pred = self.scaler_y.inverse_transform(pred_scaled.reshape(-1, 1)).flatten()
                actual = self.scaler_y.inverse_transform(actual_scaled.reshape(-1, 1)).flatten()
                
                predictions.extend(pred)
                actuals.extend(actual)
        
        test_loss /= len(test_loader)
        
        # 计算评估指标
        predictions = np.array(predictions)
        actuals = np.array(actuals)
        
        # MAE (Mean Absolute Error)
        mae = np.mean(np.abs(predictions - actuals))
        
        # RMSE (Root Mean Squared Error)
        rmse = np.sqrt(np.mean((predictions - actuals) ** 2))
        
        # MAPE (Mean Absolute Percentage Error)
        mape = np.mean(np.abs((actuals - predictions) / (actuals + 1e-8))) * 100
        
        # R² Score
        ss_res = np.sum((actuals - predictions) ** 2)
        ss_tot = np.sum((actuals - np.mean(actuals)) ** 2)
        r2 = 1 - (ss_res / (ss_tot + 1e-8))
        
        metrics = {
            'mae': mae,
            'rmse': rmse,
            'mape': mape,
            'r2': r2
        }
        
        return test_loss, metrics
    
    def predict(self, data: Optional[pd.DataFrame] = None, close_prices: Optional[np.ndarray] = None) -> float:
        """
        预测下一天的价位
        
        Args:
            data: 包含最近21天特征的数据框（如果 use_close_only=False）
            close_prices: 最近21天的 close_price 数组（如果 use_close_only=True）
        
        Returns:
            预测的价位（收盘价）
        """
        if not self.is_trained:
            raise ValueError("模型尚未训练，请先调用 train() 方法")
        
        if self.use_close_only:
            # 只使用 close_price
            if close_prices is None:
                raise ValueError("use_close_only=True 时，必须提供 close_prices 参数")
            
            if len(close_prices) < self.seq_len:
                raise ValueError(
                    f"数据不足，需要至少 {self.seq_len} 天的数据，"
                    f"当前只有 {len(close_prices)} 天"
                )
            
            # 提取最后21天的数据
            X = close_prices[-self.seq_len:].reshape(-1, 1)
            
            # 标准化
            X_scaled = self.scaler_X.transform(X)
            X_scaled = X_scaled.reshape(1, self.seq_len, 1)  # 添加 batch 维度
            
        else:
            # 使用多特征
            if data is None:
                raise ValueError("use_close_only=False 时，必须提供 data 参数")
            
            # 确保数据按日期排序
            if 'datetime' in data.columns:
                data = data.sort_values('datetime')
            elif data.index.name == 'datetime' or isinstance(data.index, pd.DatetimeIndex):
                data = data.sort_index()
            
            # 提取最后21天的数据
            if len(data) < self.seq_len:
                raise ValueError(
                    f"数据不足，需要至少 {self.seq_len} 天的数据，"
                    f"当前只有 {len(data)} 天"
                )
            
            # 选择特征列
            available_features = [f for f in self.feature_names if f in data.columns]
            if len(available_features) != len(self.feature_names):
                raise ValueError(
                    f"特征不匹配，需要: {self.feature_names}, "
                    f"可用: {available_features}"
                )
            
            # 提取最后21天的特征
            X = data[self.feature_names].tail(self.seq_len).values
            
            # 处理缺失值
            X_df = pd.DataFrame(X)
            X_df = X_df.ffill().fillna(0)
            X = X_df.values
            
            # 标准化
            n_samples, n_features = X.shape
            X_scaled = self.scaler_X.transform(X.reshape(-1, n_features))
            X_scaled = X_scaled.reshape(1, self.seq_len, n_features)  # 添加 batch 维度
        
        # 转换为张量并预测
        X_tensor = torch.FloatTensor(X_scaled)
        self.model.eval()
        with torch.no_grad():
            prediction_scaled = self.model(X_tensor).squeeze().item()
        
        # 反标准化
        prediction = self.scaler_y.inverse_transform([[prediction_scaled]])[0][0]
        
        return prediction
    
    def save_model(self):
        """保存模型和标准化器"""
        # 保存模型
        torch.save(self.model.state_dict(), self.model_path)
        logger.info(f"模型已保存到: {self.model_path}")
        
        # 保存标准化器
        scaler_data = {
            'scaler_X': self.scaler_X,
            'scaler_y': self.scaler_y,
            'feature_names': self.feature_names,
            'input_size': self.input_size,
            'hidden_size': self.hidden_size,
            'num_layers': self.num_layers,
            'dropout': self.dropout,
            'seq_len': self.seq_len,
            'use_close_only': self.use_close_only
        }
        with open(self.scaler_path, 'wb') as f:
            pickle.dump(scaler_data, f)
        logger.info(f"标准化器已保存到: {self.scaler_path}")
    
    def load_model(self):
        """加载模型和标准化器"""
        if not self.model_path.exists():
            raise FileNotFoundError(f"模型文件不存在: {self.model_path}")
        
        if not self.scaler_path.exists():
            raise FileNotFoundError(f"标准化器文件不存在: {self.scaler_path}")
        
        # 加载标准化器和配置
        with open(self.scaler_path, 'rb') as f:
            scaler_data = pickle.load(f)
        
        self.scaler_X = scaler_data['scaler_X']
        self.scaler_y = scaler_data['scaler_y']
        self.feature_names = scaler_data['feature_names']
        self.input_size = scaler_data['input_size']
        self.hidden_size = scaler_data['hidden_size']
        self.num_layers = scaler_data['num_layers']
        self.dropout = scaler_data['dropout']
        self.seq_len = scaler_data['seq_len']
        self.use_close_only = scaler_data.get('use_close_only', False)
        
        # 重新初始化模型
        self.model = LSTMPredictor(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            dropout=self.dropout
        )
        
        # 加载模型权重
        self.model.load_state_dict(torch.load(self.model_path))
        self.model.eval()
        
        self.is_trained = True
        logger.info(f"模型已从 {self.model_path} 加载")
    
    def load_data_from_db(
        self,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> pd.DataFrame:
        """
        从数据库加载股票数据
        
        Args:
            start_date: 开始日期（格式: YYYY-MM-DD），如果为 None 则加载所有数据
            end_date: 结束日期（格式: YYYY-MM-DD），如果为 None 则加载所有数据
        
        Returns:
            包含股票数据的 DataFrame
        """
        market = Market()
        data = market.get_price_data(self.stock_code, start_date, end_date)
        
        if data.empty:
            raise ValueError(f"未找到股票 {self.stock_code} 的数据")
        
        # 加载特征（如果需要）
        # 这里可以扩展为加载更多特征
        return data
    
