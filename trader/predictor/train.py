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
from trader.dataloader import dataloader_linear
from trader.logger import get_logger
from trader.config import PROJECT_ROOT
import sqlite3
from trader.config import DB_PATH

logger = get_logger(__name__)


def load_stock_data(stock_code: str, start_date: Optional[str] = None, 
                   end_date: Optional[str] = None) -> pd.DataFrame:
    """
    加载股票数据（使用 dataloader_linear，会处理缺失值和节假日）
    
    使用 dataloader_linear 可以：
    1. 自动处理缺失值（线性插值）
    2. 补全节假日数据
    3. 确保数据连续性
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
            cursor = conn.execute(query, (stock_code,))
            row = cursor.fetchone()
            conn.close()
            
            if not row or row[0] is None:
                logger.warning(f"股票 {stock_code} 没有数据")
                return pd.DataFrame()
            
            if start_date is None:
                start_date = row[0]
            if end_date is None:
                end_date = row[1]
        
        # 使用 dataloader_linear 加载数据（只加载 close_price）
        loader = dataloader_linear(stock_code)
        data = loader.load(
            start_date=start_date,
            end_date=end_date,
            feature_names=['close_price'],
            force=False
        )
        
        if data.empty:
            logger.warning(f"股票 {stock_code} 没有数据")
            return pd.DataFrame()
        
        # 确保 close_price 列存在
        if 'close_price' not in data.columns:
            logger.warning(f"股票 {stock_code} 没有 close_price 列")
            return pd.DataFrame()
        
        # 确保 close_price 是数值类型
        data['close_price'] = pd.to_numeric(data['close_price'], errors='coerce')
        
        # dataloader_linear 已经处理了缺失值，但为了安全起见，检查一下
        if data['close_price'].isna().any():
            logger.warning(f"股票 {stock_code} 仍有缺失值，使用前向和后向填充")
            data['close_price'] = data['close_price'].ffill().bfill().fillna(0)
        
        logger.info(
            f"成功从 dataloader_linear 加载 {stock_code} 的数据: "
            f"{len(data)} 条记录, "
            f"日期范围: {data.index.min()} 到 {data.index.max()}"
        )
        
        return data
        
    except Exception as e:
        logger.error(f"加载股票 {stock_code} 数据时出错: {e}", exc_info=True)
        return pd.DataFrame()


def create_sequences_from_data(data: pd.DataFrame, seq_len: int = 21, use_log_returns: bool = True):
    """
    从数据创建序列（前21天预测后1天）
    
    如果 use_log_returns=True，使用对数收益率 log(close_{t+1}/close_t)
    这样可以避免不同股票价格水平差异的问题，并且更符合金融建模习惯
    """
    # 确保数据按日期排序
    if isinstance(data.index, pd.DatetimeIndex):
        data = data.sort_index()
    elif 'datetime' in data.columns:
        data = data.sort_values('datetime')
        if not isinstance(data.index, pd.DatetimeIndex):
            data = data.set_index('datetime')
    
    close_prices = data['close_price'].values
    
    if use_log_returns:
        # 计算对数收益率：log(close_t / close_{t-1}) = log(1 + return)
        # 使用 log 可以更好地处理不同价格水平，并且对极端值更稳健
        log_returns = np.diff(np.log(close_prices + 1e-8))
        # 第一个价格作为基准，后续都是对数收益率
        sequences = []
        targets = []
        
        # 需要 seq_len+1 个价格点来创建 seq_len 个对数收益率序列
        for i in range(seq_len + 1, len(close_prices)):
            # 提取对数收益率序列
            seq_log_returns = log_returns[i - seq_len - 1:i - 1]
            sequences.append(seq_log_returns)
            # 目标值是下一天的对数收益率 log(close_{t+1}/close_t)
            targets.append(log_returns[i - 1])
        
        return np.array(sequences), np.array(targets)
    else:
        # 使用绝对价格（不推荐用于多股票训练）
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
    
    # 使用对数收益率模式（推荐用于多股票训练）
    # log(close_{t+1}/close_t) 可以避免不同股票价格水平差异，并且更稳健
    use_log_returns = True
    logger.info(f"使用对数收益率模式: {use_log_returns}（目标: log(close_{{t+1}}/close_t)）")
    
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
        
        # 确保数据按日期排序
        if 'datetime' in data.columns:
            data = data.sort_values('datetime')
        elif not isinstance(data.index, pd.DatetimeIndex):
            data = data.sort_index()
        
        # 创建序列（使用对数收益率）
        sequences, targets = create_sequences_from_data(data, predictor.seq_len, use_log_returns=use_log_returns)
        
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
    
    # 数据标准化（收益率已经相对标准化，但可以进一步标准化）
    from sklearn.preprocessing import StandardScaler
    
    X_reshaped = X_all.reshape(-1, 1)
    predictor.scaler_X = StandardScaler()
    X_scaled = predictor.scaler_X.fit_transform(X_reshaped)
    X_scaled = X_scaled.reshape(-1, predictor.seq_len, 1)
    
    predictor.scaler_y = StandardScaler()
    y_scaled = predictor.scaler_y.fit_transform(y_all.reshape(-1, 1)).flatten()
    
    # 保存使用对数收益率的标志
    predictor.use_log_returns = use_log_returns
    
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
    
    # 优化器：使用 AdamW（带权重衰减）
    learning_rate = training_config.get('learning_rate', 0.001)
    weight_decay = training_config.get('weight_decay', 1e-4)
    optimizer = optim.AdamW(
        predictor.model.parameters(), 
        lr=learning_rate,
        weight_decay=weight_decay
    )
    
    # 损失函数：使用 HuberLoss（对极端值更稳健）
    # delta=1.0 是默认值，可以根据需要调整
    criterion = nn.HuberLoss(delta=training_config.get('huber_delta', 1.0))
    
    logger.info(f"优化器: AdamW (lr={learning_rate}, weight_decay={weight_decay})")
    logger.info(f"损失函数: HuberLoss (delta={training_config.get('huber_delta', 1.0)})")
    
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
            
            # 直接使用在线模式，wandb 会自动处理登录状态
            # 如果未登录，wandb 会提示或使用离线模式
            logger.info("初始化 wandb（在线模式）...")
            
            try:
                run = wandb.init(
                    project="lstm-simple",
                    name="shared_model",
                    mode="online",  # 强制使用在线模式
                    config=wandb_config
                )
                
                # wandb 成功初始化，检查是否有 URL（在线模式会有 URL）
                if hasattr(run, 'url') and run.url:
                    logger.info("✓ wandb 在线模式已启动，数据将同步到 wandb.ai")
                    logger.info(f"  项目: lstm-simple")
                    logger.info(f"  运行: shared_model")
                    logger.info(f"  查看: {run.url}")
                else:
                    logger.info("✓ wandb 已启动（在线模式）")
                    logger.info(f"  项目: lstm-simple")
                    logger.info(f"  运行: shared_model")
                    
            except Exception as e:
                logger.error(f"wandb 在线模式初始化失败: {e}")
                logger.warning("尝试使用离线模式...")
                try:
                    run = wandb.init(
                        project="lstm-simple",
                        name="shared_model",
                        mode="offline",
                        config=wandb_config
                    )
                    logger.info(f"wandb 离线模式已启动，数据保存在: {PROJECT_ROOT / 'wandb'}")
                    logger.info(f"要同步到云端，请运行: wandb sync {PROJECT_ROOT / 'wandb' / 'offline-run-*'}")
                except Exception as e2:
                    logger.error(f"wandb 离线模式也失败: {e2}")
                    use_wandb = False
                    
        except ImportError:
            logger.warning("wandb 未安装，跳过 wandb 记录")
            logger.warning("安装命令: poetry add wandb")
            use_wandb = False
    
    # 训练循环（带早停）
    epochs = training_config.get('epochs', 50)
    verbose = training_config.get('verbose', True)
    patience = training_config.get('early_stopping_patience', 10)
    min_delta = training_config.get('early_stopping_min_delta', 1e-6)
    gradient_clip = training_config.get('gradient_clip', 1.0)
    
    best_val_loss = float('inf')
    patience_counter = 0
    
    logger.info(f"早停设置: patience={patience}, min_delta={min_delta}")
    logger.info(f"梯度裁剪: clip_norm={gradient_clip}")
    
    for epoch in range(epochs):
        # 训练阶段
        predictor.model.train()
        train_loss = 0.0
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = predictor.model(batch_X)
            loss = criterion(outputs.squeeze(), batch_y)
            loss.backward()
            
            # 梯度裁剪
            if gradient_clip > 0:
                torch.nn.utils.clip_grad_norm_(predictor.model.parameters(), gradient_clip)
            
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
        
        # 早停检查
        if val_loss < best_val_loss - min_delta:
            best_val_loss = val_loss
            patience_counter = 0
            predictor.save_model()
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logger.info(
                    f"早停触发: 验证损失在 {patience} 个 epoch 内未改善 "
                    f"(最佳: {best_val_loss:.6f}, 当前: {val_loss:.6f})"
                )
                break
        
        if verbose and (epoch + 1) % 10 == 0:
            logger.info(
                f"Epoch [{epoch+1}/{epochs}], "
                f"Train Loss: {train_loss:.6f}, "
                f"Val Loss: {val_loss:.6f}, "
                f"Best Val Loss: {best_val_loss:.6f}, "
                f"Patience: {patience_counter}/{patience}"
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
