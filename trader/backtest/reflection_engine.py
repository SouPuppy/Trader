"""
Reflection Engine：继承BacktestEngine，支持参数追踪和详细报告
"""
from typing import Dict, Optional, Callable, List
from pathlib import Path
from datetime import datetime
from trader.backtest.engine import BacktestEngine
from trader.backtest.account import Account
from trader.backtest.market import Market
from trader.backtest.report import BacktestReport
from trader.agent.theta import Theta
from trader.logger import get_logger

logger = get_logger(__name__)


class ReflectionEngine(BacktestEngine):
    """
    反思引擎：继承BacktestEngine，添加参数追踪功能
    
    新增功能：
    1. 记录每日参数θ的变化
    2. 生成参数化报告
    3. 支持反思层集成
    """
    
    def __init__(
        self,
        account: Account,
        market: Market,
        initial_theta: Theta,
        enable_report: bool = True,
        report_output_dir: Optional[Path] = None,
        report_title: Optional[str] = None,
        train_test_split_ratio: float = 0.7,
        only_test_period: bool = True,
        record_trade_history: bool = True,
        clear_trade_history: bool = True
    ):
        """
        初始化反思引擎
        
        Args:
            account: 账户实例
            market: 市场实例
            initial_theta: 初始参数θ
            enable_report: 是否启用报告生成
            report_output_dir: 报告输出目录
            report_title: 报告标题
            train_test_split_ratio: 训练/测试分割比例
            only_test_period: 是否只运行测试期
            record_trade_history: 是否记录交易历史
            clear_trade_history: 是否清空交易历史
        """
        super().__init__(
            account=account,
            market=market,
            enable_report=enable_report,
            report_output_dir=report_output_dir,
            report_title=report_title,
            train_test_split_ratio=train_test_split_ratio,
            only_test_period=only_test_period,
            record_trade_history=record_trade_history,
            clear_trade_history=clear_trade_history
        )
        
        self.initial_theta = initial_theta
        self.current_theta = initial_theta.copy()
        
        # 记录每日参数变化
        self.theta_history: List[Dict] = []
        
        # 创建参数化报告
        if enable_report:
            from trader.backtest.parametrized_report import ParametrizedReport
            self.param_report = ParametrizedReport(
                report_output_dir, title=report_title
            )
        else:
            self.param_report = None
    
    def update_theta(self, new_theta: Theta, date: Optional[str] = None):
        """
        更新参数θ
        
        Args:
            new_theta: 新的参数θ
            date: 更新日期（如果为None则使用当前日期）
        """
        self.current_theta = new_theta
        update_date = date or self.current_date or datetime.now().strftime("%Y-%m-%d")
        
        # 记录参数变化
        self.theta_history.append({
            "date": update_date,
            "theta": new_theta.to_dict()
        })
        
        logger.info(f"[ReflectionEngine] 更新参数θ on {update_date}: {new_theta}")
    
    def get_current_theta(self) -> Theta:
        """获取当前参数θ"""
        return self.current_theta
    
    def get_theta_history(self) -> List[Dict]:
        """获取参数θ历史"""
        return self.theta_history.copy()
    
    def record_daily_theta(self, date: Optional[str] = None):
        """
        记录每日参数θ（即使没有变化也记录）
        
        Args:
            date: 日期（如果为None则使用当前日期）
        """
        record_date = date or self.current_date or datetime.now().strftime("%Y-%m-%d")
        
        # 检查今天是否已经记录过
        if self.theta_history:
            last_record = self.theta_history[-1]
            if last_record["date"] == record_date:
                return  # 已经记录过
        
        # 记录当前参数
        self.theta_history.append({
            "date": record_date,
            "theta": self.current_theta.to_dict()
        })
    
    def generate_parametrized_report(
        self,
        stock_codes: List[str],
        start_date: str,
        end_date: str,
        strategy_name: Optional[str] = None,
        is_single_stock: bool = False
    ) -> Path:
        """
        生成参数化报告
        
        Args:
            stock_codes: 股票代码列表
            start_date: 开始日期
            end_date: 结束日期
            strategy_name: 策略名称
            is_single_stock: 是否为单股票模式
            
        Returns:
            Path: 报告文件路径
        """
        if not self.param_report:
            raise ValueError("参数化报告未启用")
        
        # 设置报告数据
        if self.report:
            self.param_report.daily_records = self.report.daily_records
        else:
            self.param_report.daily_records = []
        
        self.param_report.train_test_split_date = self.train_test_split_date
        self.param_report.theta_history = self.theta_history.copy()
        self.param_report.initial_theta = self.initial_theta.to_dict()
        self.param_report.final_theta = self.current_theta.to_dict()
        
        # 生成报告
        return self.param_report.generate_parametrized_report(
            account=self.account,
            stock_codes=stock_codes,
            start_date=start_date,
            end_date=end_date,
            strategy_name=strategy_name,
            is_single_stock=is_single_stock
        )



