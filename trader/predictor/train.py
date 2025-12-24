"""
简单的 LSTM 模型训练脚本
用前21天的 close_price 预测后一天的 close_price
用10支股票训练一个共享参数的模型，10支股票测试
"""
import sys
from pathlib import Path
from typing import List, Optional
import pandas as pd
import numpy as np

# 添加项目根目录到 Python 路径
project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from trader.predictor.Predictor import Predictor
from trader.predictor.config_loader import (
    get_train_stocks,
    get_test_stocks,
    get_model_config,
    get_training_config,
    get_data_config
)
from trader.backtest.market import Market
from trader.logger import get_logger

logger = get_logger(__name__)


def load_stock_data(stock_code: str, start_date: Optional[str] = None, 
                   end_date: Optional[str] = None) -> pd.DataFrame:
    """加载股票数据"""
    market = Market()
    data = market.get_price_data(stock_code, start_date, end_date)
    
    if data.empty:
        logger.warning(f"股票 {stock_code} 没有数据")
        return pd.DataFrame()
    
    return data


def create_sequences_from_data(data: pd.DataFrame, seq_len: int = 21):
    """从数据创建序列（前21天预测后1天）"""
    close_prices = data['close_price'].values
    
    sequences = []
    targets = []
    
    for i in range(seq_len, len(close_prices)):
        sequences.append(close_prices[i - seq_len:i])
        targets.append(close_prices[i])
    
    return np.array(sequences), np.array(targets)


def train_shared_model(
    train_stocks: List[str],
    model_config: dict,
    training_config: dict,
    data_config: dict,
    use_wandb: bool = False
) -> Predictor:
    """训练一个共享参数的模型，使用所有训练股票的数据"""
    logger.info(f"开始训练共享 LSTM 模型（使用 {len(train_stocks)} 支股票的数据）")
    
    # 创建共享预测器
    predictor = Predictor(
        stock_code="shared",
        hidden_size=model_config.get('hidden_size', 64),
        num_layers=model_config.get('num_layers', 2),
        dropout=model_config.get('dropout', 0.2),
        seq_len=model_config.get('seq_len', 21),
        use_close_only=True  # 只用 close_price
    )
    
    # 加载所有股票的数据并创建序列
    start_date = data_config.get('start_date') or None
    end_date = data_config.get('end_date') or None
    
    all_sequences = []
    all_targets = []
    
    for stock_code in train_stocks:
        logger.info(f"加载股票数据: {stock_code}")
        data = load_stock_data(stock_code, start_date, end_date)
        
        if data.empty:
            logger.warning(f"股票 {stock_code} 没有数据，跳过")
            continue
        
        # 确保有 close_price 列
        if 'close_price' not in data.columns:
            logger.warning(f"股票 {stock_code} 没有 close_price 列，跳过")
            continue
        
        # 创建序列
        sequences, targets = create_sequences_from_data(data, predictor.seq_len)
        
        if len(sequences) > 0:
            all_sequences.append(sequences)
            all_targets.append(targets)
            logger.info(f"  股票 {stock_code}: {len(sequences)} 个序列")
        else:
            logger.warning(f"  股票 {stock_code}: 数据不足，无法创建序列")
    
    if len(all_sequences) == 0:
        raise ValueError("没有找到任何可用的训练数据")
    
    # 合并所有序列
    X_all = np.concatenate(all_sequences, axis=0)
    y_all = np.concatenate(all_targets, axis=0)
    
    logger.info(f"合并后的总序列数: {len(X_all)}")
    
    # 重塑为 LSTM 输入格式 (n_samples, seq_len, n_features)
    X_all = X_all.reshape(-1, predictor.seq_len, 1)
    
    # 数据标准化（全局标准化）
    from sklearn.preprocessing import StandardScaler
    
    X_reshaped = X_all.reshape(-1, 1)
    predictor.scaler_X = StandardScaler()
    X_scaled = predictor.scaler_X.fit_transform(X_reshaped)
    X_scaled = X_scaled.reshape(-1, predictor.seq_len, 1)
    
    predictor.scaler_y = StandardScaler()
    y_scaled = predictor.scaler_y.fit_transform(y_all.reshape(-1, 1)).flatten()
    
    # 分割训练集和验证集
    validation_split = training_config.get('validation_split', 0.2)
    val_split_idx = int(len(X_scaled) * (1 - validation_split))
    
    X_train = X_scaled[:val_split_idx]
    y_train = y_scaled[:val_split_idx]
    X_val = X_scaled[val_split_idx:]
    y_val = y_scaled[val_split_idx:]
    
    logger.info(f"训练集: {len(X_train)} 个样本，验证集: {len(X_val)} 个样本")
    
    # 训练模型
    from torch.utils.data import Dataset, DataLoader
    import torch
    import torch.nn as nn
    import torch.optim as optim
    
    # 创建数据集
    class StockDataset(Dataset):
        def __init__(self, sequences, targets):
            self.sequences = torch.FloatTensor(sequences)
            self.targets = torch.FloatTensor(targets)
        
        def __len__(self):
            return len(self.sequences)
        
        def __getitem__(self, idx):
            return self.sequences[idx], self.targets[idx]
    
    train_dataset = StockDataset(X_train, y_train)
    val_dataset = StockDataset(X_val, y_val)
    
    train_loader = DataLoader(train_dataset, batch_size=training_config.get('batch_size', 32), shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=training_config.get('batch_size', 32), shuffle=False)
    
    # 优化器和损失函数
    optimizer = optim.Adam(predictor.model.parameters(), lr=training_config.get('learning_rate', 0.001))
    criterion = nn.MSELoss()
    
    # 初始化 wandb
    if use_wandb:
        try:
            import wandb
            wandb_config = {
                "model_type": "shared",
                "num_stocks": len(train_stocks),
                "stocks": train_stocks,
                "hidden_size": predictor.hidden_size,
                "num_layers": predictor.num_layers,
                "dropout": predictor.dropout,
                "seq_len": predictor.seq_len,
                "epochs": training_config.get('epochs', 50),
                "batch_size": training_config.get('batch_size', 32),
                "learning_rate": training_config.get('learning_rate', 0.001),
            }
            
            try:
                api = wandb.Api()
                try:
                    viewer = wandb.api.viewer()
                    wandb_mode = "online" if viewer else "offline"
                except:
                    wandb_mode = "offline"
            except:
                wandb_mode = "offline"
            
            wandb.init(
                project="lstm-simple",
                name="shared_model",
                mode=wandb_mode,
                config=wandb_config
            )
        except ImportError:
            logger.warning("wandb 未安装，跳过 wandb 记录")
            use_wandb = False
    
    # 训练循环
    epochs = training_config.get('epochs', 50)
    verbose = training_config.get('verbose', True)
    best_val_loss = float('inf')
    
    for epoch in range(epochs):
        # 训练阶段
        predictor.model.train()
        train_loss = 0.0
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = predictor.model(batch_X)
            loss = criterion(outputs.squeeze(), batch_y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        
        # 验证阶段
        predictor.model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                outputs = predictor.model(batch_X)
                loss = criterion(outputs.squeeze(), batch_y)
                val_loss += loss.item()
        
        val_loss /= len(val_loader)
        
        # 记录到 wandb
        if use_wandb:
            try:
                import wandb
                wandb.log({
                    "epoch": epoch + 1,
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                })
            except:
                pass
        
        # 保存最佳模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            predictor.save_model()
        
        if verbose and (epoch + 1) % 10 == 0:
            logger.info(
                f"Epoch [{epoch+1}/{epochs}], "
                f"Train Loss: {train_loss:.6f}, "
                f"Val Loss: {val_loss:.6f}"
            )
    
    predictor.is_trained = True
    logger.info(f"训练完成，最佳验证损失: {best_val_loss:.6f}")
    
    if use_wandb:
        try:
            import wandb
            wandb.finish()
        except:
            pass
    
    return predictor


def train_all_stocks(config_path: Optional[Path] = None, use_wandb: bool = False):
    """训练共享参数的模型"""
    logger.info("开始训练共享 LSTM 模型（简单模式：前21天预测后1天）")
    
    # 加载配置
    train_stocks = get_train_stocks(config_path)
    test_stocks = get_test_stocks(config_path)
    model_config = get_model_config(config_path)
    training_config = get_training_config(config_path)
    data_config = get_data_config(config_path)
    
    logger.info(f"训练集股票数量: {len(train_stocks)}")
    logger.info(f"训练集股票: {', '.join(train_stocks)}")
    logger.info(f"测试集股票数量: {len(test_stocks)}")
    logger.info(f"测试集股票: {', '.join(test_stocks)}")
    
    # 训练共享模型
    predictor = train_shared_model(
        train_stocks=train_stocks,
        model_config=model_config,
        training_config=training_config,
        data_config=data_config,
        use_wandb=use_wandb
    )
    
    # 总结
    logger.info("\n训练总结")
    logger.info(f"成功训练共享模型，使用 {len(train_stocks)} 支股票的数据")
    from trader.config import PROJECT_ROOT
    logger.info(f"模型保存位置: {predictor.model_path}")
    logger.info(f"标准化器保存位置: {predictor.scaler_path}")


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="训练共享参数的 LSTM 股票预测模型（前21天预测后1天）")
    parser.add_argument(
        '--config',
        type=str,
        default=None,
        help='配置文件路径（默认使用 model_config.toml）'
    )
    parser.add_argument(
        '--wandb',
        action='store_true',
        help='使用 wandb 记录训练过程'
    )
    
    args = parser.parse_args()
    
    config_path = Path(args.config) if args.config else None
    
    # 训练共享模型
    train_all_stocks(
        config_path=config_path,
        use_wandb=args.wandb
    )
    
    logger.info("\n训练完成！")


if __name__ == '__main__':
    main()
