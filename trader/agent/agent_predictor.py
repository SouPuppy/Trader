"""
Predictor Agent - 使用 LSTM Predictor 预测价格的交易策略 Agent
使用训练好的 Predictor 模型来预测股票价格，基于预测价格变化计算 score
"""
import sys
from pathlib import Path
from typing import Dict, Optional
import numpy as np
import pandas as pd

# 添加项目根目录到 Python 路径
project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from trader.agent.TradingAgent import TradingAgent
from trader.backtest.engine import BacktestEngine
from trader.backtest.market import Market
from trader.predictor.Predictor import Predictor
from trader.predictor.config_loader import get_model_config
from trader.logger import get_logger

logger = get_logger(__name__)


class PredictorAgent(TradingAgent):
    """
    使用 LSTM Predictor 预测价格的交易策略 Agent
    
    核心逻辑：
    1. 使用训练好的 Predictor 模型预测下一天的价位
    2. 基于预测价格和当前价格的差异计算 score
    3. 预测价格上涨 -> 正 score，预测价格下跌 -> 负 score
    """
    
    def __init__(
        self,
        name: str = "PredictorAgent",
        stock_code: str = None,  # 已废弃，保留以兼容性
        max_position_weight: float = 0.1,
        min_score_threshold: float = 0.0,
        max_total_weight: float = 1.0,
        model_path: Optional[Path] = None,
        use_close_only: bool = True,
        seq_len: int = 21,
        use_square_weight: bool = False,  # 是否使用平方映射（让高score获得更多仓位）
        debug: bool = False  # 是否输出详细调试信息
    ):
        """
        初始化 Predictor Agent
        
        注意：使用共享模型（shared model），对所有股票都适用
        
        Args:
            name: Agent 名称
            stock_code: 已废弃，保留以兼容性（实际使用共享模型）
            max_position_weight: 单个股票最大配置比例
            min_score_threshold: 最小 score 阈值
            max_total_weight: 总配置比例上限
            model_path: 模型路径，如果为 None 则使用默认共享模型路径
            use_close_only: 是否只使用 close_price（必须与训练时一致）
            seq_len: 序列长度（默认21天，必须与训练时一致）
            use_square_weight: 是否使用平方映射（让高score获得更多仓位）
        """
        super().__init__(
            name=name,
            max_position_weight=max_position_weight,
            min_score_threshold=min_score_threshold,
            max_total_weight=max_total_weight
        )
        
        self.use_close_only = use_close_only
        self.seq_len = seq_len
        self.use_square_weight = use_square_weight
        self.debug = debug
        
        # 加载模型配置
        model_config = get_model_config()
        self.hidden_size = model_config.get('hidden_size', 64)
        self.num_layers = model_config.get('num_layers', 2)
        self.dropout = model_config.get('dropout', 0.2)
        
        # 初始化 Predictor（延迟加载共享模型）
        self.predictor: Optional[Predictor] = None
        self.model_path = model_path
        self._model_loaded = False
        
        # 缓存最近的价格数据（避免重复获取）
        self._price_cache: Dict[str, np.ndarray] = {}
    
    def _load_predictor(self):
        """
        加载共享 Predictor 模型（延迟加载，只加载一次）
        
        使用 stock_code="shared" 来加载共享模型，对所有股票都适用
        """
        if self._model_loaded and self.predictor is not None:
            return
        
        try:
            logger.info("加载共享 Predictor 模型 (shared model)")
            logger.debug(f"模型配置: hidden_size={self.hidden_size}, num_layers={self.num_layers}, "
                        f"dropout={self.dropout}, seq_len={self.seq_len}, "
                        f"use_close_only={self.use_close_only}")
            
            # 创建 Predictor 实例，使用共享模型
            self.predictor = Predictor(
                stock_code="shared",  # 使用共享模型
                hidden_size=self.hidden_size,
                num_layers=self.num_layers,
                dropout=self.dropout,
                seq_len=self.seq_len,
                model_path=self.model_path,
                use_close_only=self.use_close_only
            )
            
            # 加载模型
            logger.debug(f"尝试加载模型: {self.predictor.model_path}")
            logger.debug(f"尝试加载标准化器: {self.predictor.scaler_path}")
            self.predictor.load_model()
            
            self._model_loaded = True
            logger.info("共享 Predictor 模型加载成功")
            logger.debug(f"模型已训练: {self.predictor.is_trained}, "
                        f"use_log_returns: {getattr(self.predictor, 'use_log_returns', False)}")
            
        except FileNotFoundError as e:
            logger.error(f"共享模型文件不存在: {e}")
            logger.error(f"请确保共享模型已训练并保存在 weights/LSTM/model.pth")
            logger.error(f"模型路径: {self.model_path if self.model_path else '默认路径'}")
            self.predictor = None
            self._model_loaded = False
        except Exception as e:
            logger.error(f"加载共享 Predictor 模型失败: {e}", exc_info=True)
            self.predictor = None
            self._model_loaded = False
    
    def _get_recent_prices(
        self,
        stock_code: str,
        engine: BacktestEngine,
        lookback_days: int = None
    ) -> Optional[np.ndarray]:
        """
        获取最近N天的收盘价
        
        Args:
            stock_code: 股票代码
            engine: 回测引擎
            lookback_days: 回看天数，如果为 None 则使用 seq_len
            
        Returns:
            最近N天的收盘价数组，如果数据不足则返回 None
        """
        if lookback_days is None:
            lookback_days = self.seq_len
        
        # 检查缓存
        cache_key = f"{stock_code}_{engine.current_date}"
        if cache_key in self._price_cache:
            prices = self._price_cache[cache_key]
            if len(prices) >= lookback_days:
                return prices[-lookback_days:]
        
        # 从市场获取价格数据（使用更高效的方法）
        try:
            market = engine.market
            current_date = engine.current_date
            
            # 使用 get_price_data 一次性获取历史数据
            # 获取从当前日期往前 lookback_days 天的数据
            # 需要计算开始日期（假设每个交易日都有数据，实际可能需要更多天数）
            price_data = market.get_price_data(
                stock_code,
                end_date=current_date
            )
            
            if price_data.empty or 'close_price' not in price_data.columns:
                logger.debug(f"未找到股票 {stock_code} 的价格数据")
                return None
            
            # 确保数据按日期排序
            if 'datetime' in price_data.columns:
                price_data = price_data.sort_values('datetime')
                # 找到当前日期对应的索引
                current_date_dt = pd.to_datetime(current_date)
                date_mask = price_data['datetime'] <= current_date_dt
                price_data = price_data[date_mask]
            
            # 获取最近 lookback_days 天的收盘价
            if len(price_data) < lookback_days:
                logger.debug(
                    f"[{stock_code}] 数据不足: 需要 {lookback_days} 天，当前只有 {len(price_data)} 天"
                )
                return None
            
            # 提取最近 lookback_days 天的收盘价
            prices = price_data['close_price'].tail(lookback_days).values
            
            # 应用价格调整系数
            prices = prices * market.price_adjustment
            
            prices_array = np.array(prices)
            
            logger.debug(
                f"[{stock_code}] 成功获取 {len(prices_array)} 天价格数据: "
                f"日期范围 [{price_data.index[-lookback_days] if hasattr(price_data, 'index') else 'N/A'}, "
                f"{price_data.index[-1] if hasattr(price_data, 'index') else 'N/A'}], "
                f"价格范围 [{prices_array.min():.2f}, {prices_array.max():.2f}], "
                f"最新价格: {prices_array[-1]:.2f}"
            )
            
            # 更新缓存
            self._price_cache[cache_key] = prices_array
            
            return prices_array
            
        except Exception as e:
            logger.error(f"获取最近价格失败: {e}", exc_info=True)
            return None
    
    def score(self, stock_code: str, engine: BacktestEngine) -> float:
        """
        计算股票的 score（使用共享 Predictor 模型预测价格）
        
        Args:
            stock_code: 股票代码
            engine: 回测引擎
            
        Returns:
            float: 分数，范围 [-1, 1]
                  - 正数表示预测价格上涨
                  - 负数表示预测价格下跌
                  - 绝对值表示预测变化幅度
        """
        # 延迟加载共享模型（只加载一次）
        if not self._model_loaded or self.predictor is None:
            if self.debug:
                logger.debug(f"[{stock_code}] 首次调用，加载模型...")
            self._load_predictor()
        
        # 如果模型加载失败，返回 0
        if self.predictor is None:
            if self.debug:
                logger.debug(f"[{stock_code}] 模型未加载，返回 score=0.0")
            else:
                logger.warning(f"[{stock_code}] 模型未加载，无法计算 score")
            return 0.0
        
        if not self.predictor.is_trained:
            if self.debug:
                logger.debug(f"[{stock_code}] 模型未训练，返回 score=0.0")
            else:
                logger.warning(f"[{stock_code}] 模型未训练，无法计算 score")
            return 0.0
        
        # 获取最近的价格数据
        if self.use_close_only:
            # 需要 seq_len 天的 close_price（如果使用对数收益率，需要 seq_len+1 天）
            use_log_returns = getattr(self.predictor, 'use_log_returns', False)
            lookback_days = self.seq_len + 1 if use_log_returns else self.seq_len
            
            if self.debug:
                logger.debug(f"[{stock_code}] 需要 {lookback_days} 天的历史数据 (seq_len={self.seq_len}, "
                            f"use_log_returns={use_log_returns})")
            
            close_prices = self._get_recent_prices(stock_code, engine, lookback_days)
            
            if close_prices is None:
                if self.debug:
                    logger.debug(f"[{stock_code}] 无法获取足够的历史价格数据，返回 score=0.0")
                else:
                    logger.warning(f"[{stock_code}] 数据不足（需要 {lookback_days} 天），无法计算 score")
                return 0.0
            
            if self.debug:
                logger.debug(f"[{stock_code}] 成功获取 {len(close_prices)} 天的价格数据，"
                            f"价格范围: [{close_prices.min():.2f}, {close_prices.max():.2f}]")
            
            try:
                # 使用 Predictor 预测下一天的价位
                if self.debug:
                    logger.debug(f"[{stock_code}] 开始预测...")
                predicted_price = self.predictor.predict(close_prices=close_prices)
                if self.debug:
                    logger.debug(f"[{stock_code}] 预测完成: predicted_price={predicted_price:.4f}")
                
            except Exception as e:
                if self.debug:
                    logger.debug(f"[{stock_code}] 预测失败: {e}", exc_info=True)
                else:
                    logger.warning(f"[{stock_code}] 预测失败: {e}")
                return 0.0
        else:
            # 使用多特征模式（需要从 engine 获取特征数据）
            logger.warning(f"[{stock_code}] 多特征模式暂未实现，请使用 use_close_only=True")
            return 0.0
        
        # 获取当前价格
        current_price = close_prices[-1]
        
        if self.debug:
            logger.debug(f"[{stock_code}] 当前价格: {current_price:.4f}, 预测价格: {predicted_price:.4f}")
        
        if current_price <= 0:
            if self.debug:
                logger.debug(f"[{stock_code}] 当前价格无效 (<=0): {current_price}, 返回 score=0.0")
            else:
                logger.warning(f"[{stock_code}] 当前价格无效 (<=0): {current_price}")
            return 0.0
        
        # 计算预测收益率
        predicted_return = (predicted_price - current_price) / current_price
        
        if self.debug:
            logger.debug(f"[{stock_code}] 预测收益率: {predicted_return*100:.4f}% "
                        f"({predicted_price:.4f} - {current_price:.4f}) / {current_price:.4f}")
        
        # 将收益率映射到 score [-1, 1]
        # 使用 tanh 函数将收益率映射到 [-1, 1] 范围
        # 由于预测收益率通常很小（0.03%-0.07%），需要更大的缩放因子
        # 使用缩放因子 50，使得 0.1% 的收益率映射到约 0.05 的 score
        # 这样即使预测收益率很小，也能产生有意义的交易信号
        score = np.tanh(predicted_return * 50.0)
        
        if self.debug:
            logger.debug(f"[{stock_code}] 最终 score: {score:.6f} (tanh({predicted_return*5.0:.6f}))")
        else:
            # 即使不在调试模式，也记录关键信息（如果 score 不为 0）
            if abs(score) > 0.001:  # 只记录有意义的 score
                logger.info(f"[{stock_code}] score={score:.6f} "
                          f"(当前价={current_price:.2f}, 预测价={predicted_price:.2f}, "
                          f"收益率={predicted_return*100:.2f}%)")
        
        return float(score)
    
    def weight(self, stock_code: str, score: float, engine: BacktestEngine) -> float:
        """
        计算实际资金配置比例（重写基类方法）
        
        使用线性映射或平方映射：
        - 线性映射：weight = score * max_position_weight
        - 平方映射：weight = score^2 * max_position_weight（让高score获得更多仓位）
        只考虑正分（预测上涨）的股票
        
        Args:
            stock_code: 股票代码
            score: 该股票的 score 值（范围 [-1, 1]）
            engine: 回测引擎
            
        Returns:
            float: 资金配置比例，范围 [0, max_position_weight]
        """
        if self.debug:
            logger.debug(f"[{stock_code}] weight() 计算: score={score:.6f}, "
                        f"min_score_threshold={self.min_score_threshold}, "
                        f"max_position_weight={self.max_position_weight}, "
                        f"use_square_weight={self.use_square_weight}")
        
        # 如果 score 低于阈值，不配置
        if score < self.min_score_threshold:
            if self.debug:
                logger.debug(f"[{stock_code}] score ({score:.6f}) < min_score_threshold "
                            f"({self.min_score_threshold}), 返回 weight=0.0")
            return 0.0
        
        # 只考虑正分（预测上涨）
        normalized_score = max(0.0, score)
        
        if self.debug:
            logger.debug(f"[{stock_code}] normalized_score (max(0, score)): {normalized_score:.6f}")
        
        # 根据配置选择线性映射或平方映射
        if self.use_square_weight:
            # 平方映射：让高score的股票获得更多仓位
            # 例如：score=0.5 -> weight=0.25*max, score=0.8 -> weight=0.64*max
            weight = (normalized_score ** 2) * self.max_position_weight
            if self.debug:
                logger.debug(f"[{stock_code}] 平方映射: weight = ({normalized_score:.6f}^2) * "
                            f"{self.max_position_weight} = {weight:.6f}")
        else:
            # 线性映射到 [0, max_position_weight]
            weight = normalized_score * self.max_position_weight
            if self.debug:
                logger.debug(f"[{stock_code}] 线性映射: weight = {normalized_score:.6f} * "
                            f"{self.max_position_weight} = {weight:.6f}")
        
        final_weight = min(weight, self.max_position_weight)
        
        if self.debug:
            logger.debug(f"[{stock_code}] 最终 weight: {final_weight:.6f} "
                        f"(min({weight:.6f}, {self.max_position_weight}))")
        elif final_weight > 0:
            # 即使不在调试模式，也记录有意义的 weight
            logger.info(f"[{stock_code}] weight={final_weight:.6f} "
                       f"(score={score:.6f}, normalized_score={normalized_score:.6f})")
        
        return final_weight
    
    def on_date(self, engine: BacktestEngine, date: str):
        """
        每个交易日的回调函数
        
        Args:
            engine: 回测引擎
            date: 当前日期
        """
        # 清理缓存（只保留最近的数据）
        if len(self._price_cache) > 10:
            # 保留最近的10个缓存
            keys_to_remove = list(self._price_cache.keys())[:-10]
            for key in keys_to_remove:
                del self._price_cache[key]

