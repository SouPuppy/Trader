"""
参数化报告：支持参数θ追踪和可视化
"""
from typing import Dict, Optional, List
from pathlib import Path
from trader.backtest.report import BacktestReport
from trader.logger import get_logger

logger = get_logger(__name__)


class ParametrizedReport(BacktestReport):
    """
    参数化报告：支持参数θ追踪和可视化
    """
    
    def __init__(self, output_dir: Optional[Path] = None, title: Optional[str] = None):
        """
        初始化参数化报告
        
        Args:
            output_dir: 输出目录
            title: 报告标题
        """
        super().__init__(output_dir, title)
        
        # 参数历史
        self.theta_history: List[Dict] = []
        self.initial_theta: Optional[Dict] = None
        self.final_theta: Optional[Dict] = None
    
    def generate_parametrized_report(
        self,
        account,
        stock_codes: List[str],
        start_date: str,
        end_date: str,
        strategy_name: Optional[str] = None,
        is_single_stock: bool = False
    ) -> Path:
        """
        生成参数化报告（支持多资产和单股票）
        
        Args:
            account: 账户实例
            stock_codes: 股票代码列表
            start_date: 开始日期
            end_date: 结束日期
            strategy_name: 策略名称
            is_single_stock: 是否为单股票模式
            
        Returns:
            Path: 报告文件路径
        """
        if is_single_stock:
            logger.info("生成参数化单股票报告...")
            template_name = "parametrized_single_stock_template.md.j2"
        else:
            logger.info("生成参数化多资产报告...")
            template_name = "parametrized_multi_asset_template.md.j2"
        
        # 准备模板数据
        template_data = self._prepare_template_data(
            account, stock_codes, start_date, end_date, strategy_name, is_single_stock
        )
        
        # 渲染模板
        template = self.jinja_env.get_template(template_name)
        content = template.render(**template_data)
        
        # 保存报告
        report_file = self.output_dir / "report.md"
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(content)
        
        logger.info(f"参数化报告已保存: {report_file}")
        
        # 生成参数变化图表
        self._generate_theta_charts()
        
        return report_file
    
    def _prepare_template_data(
        self,
        account,
        stock_codes: List[str],
        start_date: str,
        end_date: str,
        strategy_name: Optional[str] = None,
        is_single_stock: bool = False
    ) -> Dict:
        """准备模板数据（包含所有多资产报告的数据分析）"""
        from datetime import datetime
        import pandas as pd
        
        # 复用父类的数据准备逻辑
        # 获取最终市场价格
        market_prices = {}
        for stock_code in stock_codes:
            # 从 daily_records 中获取最后的价格
            if self.daily_records:
                last_record = self.daily_records[-1]
                if stock_code in last_record.get('positions', {}):
                    market_prices[stock_code] = last_record['positions'][stock_code]['current_price']
        
        # 如果没有价格，使用账户的持仓价格
        if not market_prices:
            for stock_code, position in account.positions.items():
                if stock_code in stock_codes:
                    # 使用平均价格作为当前价格（如果没有更好的数据）
                    market_prices[stock_code] = position['average_price']
        
        # 计算最终状态
        final_equity = account.equity(market_prices)
        final_profit = account.get_total_profit(market_prices)
        final_return_pct = account.get_total_return(market_prices)
        
        # 准备最终状态数据
        final_state = {
            'cash': account.cash,
            'positions_value': final_equity - account.cash,
            'equity': final_equity,
            'profit': final_profit,
            'return_pct': final_return_pct,
            'positions': {}
        }
        
        # 添加持仓明细
        for stock_code in stock_codes:
            position = account.get_position(stock_code)
            if position and stock_code in market_prices:
                price = market_prices[stock_code]
                shares = position['shares']
                avg_price = position['average_price']
                value = shares * price
                profit = (price - avg_price) * shares
                return_pct = ((price - avg_price) / avg_price * 100) if avg_price > 0 else 0.0
                weight = value / final_equity if final_equity > 0 else 0.0
                
                final_state['positions'][stock_code] = {
                    'shares': shares,
                    'average_price': avg_price,
                    'current_price': price,
                    'market_value': value,
                    'profit': profit,
                    'return_pct': return_pct,
                    'weight': weight
                }
        
        # 计算各股票表现和贡献度
        stock_performance = {}
        for stock_code in stock_codes:
            if stock_code in final_state['positions']:
                pos = final_state['positions'][stock_code]
                # 计算初始权重（从交易记录中估算）
                initial_weight = 0.0
                buy_trades = [t for t in account.trades if t['stock_code'] == stock_code and t['type'] == 'buy']
                if buy_trades:
                    total_buy_cost = sum(t['cost'] for t in buy_trades)
                    initial_weight = total_buy_cost / account.initial_cash if account.initial_cash > 0 else 0.0
                
                # 计算贡献度（权重 * 收益率）
                contribution = pos['weight'] * pos['return_pct']
                
                # 计算交易次数
                num_trades = len([t for t in account.trades if t['stock_code'] == stock_code])
                
                stock_performance[stock_code] = {
                    'initial_weight': initial_weight,
                    'final_weight': pos['weight'],
                    'return_pct': pos['return_pct'],
                    'profit': pos['profit'],
                    'num_trades': num_trades,
                    'contribution': contribution
                }
        
        # 准备模板数据（基础部分）
        template_data = {
            'title': self.title,
            'strategy_name': strategy_name or self.title,
            'start_date': start_date,
            'end_date': end_date,
            'trading_days': len(self.daily_records),
            'stock_count': len(stock_codes),
            'stock_code': stock_codes[0] if stock_codes else '',  # 单股票模板使用
            'initial_cash': account.initial_cash,
            'final_state': final_state,
            'stock_performance': stock_performance,
            'strategy_params': {},  # 可以添加策略参数
            'report_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            # 参数追踪数据
            'initial_theta': self.initial_theta or {},
            'final_theta': self.final_theta or {},
            'theta_history': self.theta_history,
            'daily_records': self.daily_records
        }
        
        # 添加交易统计
        if account.trades:
            buy_trades = [t for t in account.trades if t['type'] == 'buy']
            sell_trades = [t for t in account.trades if t['type'] == 'sell']
            template_data['trade_statistics'] = {
                'total_trades': len(account.trades),
                'buy_count': len(buy_trades),
                'sell_count': len(sell_trades),
                'total_buy_cost': sum(t['cost'] for t in buy_trades),
                'total_sell_revenue': sum(t.get('revenue', 0) for t in sell_trades),
                'realized_profit': sum(t.get('profit', 0) for t in sell_trades)
            }
            
            # 添加交易记录（按日期排序）
            sorted_trades = sorted(
                account.trades,
                key=lambda t: t['date'] if isinstance(t['date'], datetime) else pd.to_datetime(t['date'])
            )
            template_data['trades'] = sorted_trades
        
        # 添加统计信息
        if len(self.daily_records) > 1:
            equities = [r['equity'] for r in self.daily_records]
            returns = [r['return_pct'] for r in self.daily_records]
            max_drawdown = self._calculate_max_drawdown(equities)
            
            daily_returns, sharpe_info = self._calculate_daily_returns(risk_free_rate_annual=0.0)
            
            # 基础统计
            template_data['statistics'] = {
                'max_equity': max(equities),
                'min_equity': min(equities),
                'max_return_pct': max(returns),
                'min_return_pct': min(returns),
                'max_drawdown_pct': max_drawdown
            }
            
            # 添加夏普比率
            if sharpe_info:
                template_data['statistics']['sharpe_annual'] = sharpe_info['sharpe_annual']
                template_data['statistics']['sharpe_daily'] = sharpe_info['sharpe_daily']
            
            # 计算 CAGR
            start_equity = equities[0]
            end_equity = equities[-1]
            cagr = self._calculate_cagr(start_equity, end_equity, len(self.daily_records))
            template_data['statistics']['cagr'] = cagr
            
            # 计算 Calmar 比率
            calmar = self._calculate_calmar(cagr, max_drawdown)
            template_data['statistics']['calmar'] = calmar
            
            # 计算波动率
            if daily_returns:
                volatility = self._calculate_volatility(daily_returns)
                template_data['statistics']['volatility_annual'] = volatility
            
            # 计算 Sortino 比率
            if daily_returns:
                sortino_info = self._calculate_sortino(daily_returns, risk_free_rate_annual=0.0)
                if sortino_info:
                    template_data['statistics']['sortino_annual'] = sortino_info.get('sortino_annual', 0.0)
                    template_data['statistics']['sortino_daily'] = sortino_info.get('sortino_daily', 0.0)
            
            # 计算尾部风险
            if daily_returns:
                var_cvar = self._calculate_tail_risk(daily_returns)
                if var_cvar:
                    template_data['statistics'].update(var_cvar)
        
        # 添加交易质量指标
        if account.trades:
            trade_metrics = self._calculate_trade_metrics(account)
            if trade_metrics:
                template_data['trade_metrics'] = trade_metrics
        
        # 添加执行指标
        if len(self.daily_records) > 0:
            execution_metrics = self._calculate_execution_metrics(account, len(self.daily_records))
            if execution_metrics:
                template_data['execution_metrics'] = execution_metrics
        
        # 生成走势图（多资产图表）
        if stock_codes:
            chart_file = self._generate_charts(
                stock_code=stock_codes[0] if stock_codes else None,
                start_date=start_date,
                end_date=end_date,
                all_stock_codes=stock_codes
            )
            if chart_file:
                # 计算相对路径
                chart_relative_path = chart_file.relative_to(self.output_dir)
                template_data['chart_file'] = str(chart_relative_path)
        
        return template_data
    
    def _generate_theta_charts(self):
        """生成参数变化图表"""
        if not self.theta_history:
            return
        
        try:
            import matplotlib.pyplot as plt
            import pandas as pd
            from datetime import datetime
            
            # 准备数据
            dates = [h["date"] for h in self.theta_history]
            theta_data = [h["theta"] for h in self.theta_history]
            
            # 创建图表
            fig, axes = plt.subplots(3, 2, figsize=(15, 12))
            fig.suptitle('Theta Parameters Evolution', fontsize=16)
            
            # 1. gross_exposure
            ax = axes[0, 0]
            values = [t["gross_exposure"] for t in theta_data]
            ax.plot(range(len(dates)), values, marker='o', linewidth=2)
            ax.set_title('Gross Exposure')
            ax.set_ylabel('Ratio')
            ax.set_xlabel('Update Count')
            ax.grid(True, alpha=0.3)
            
            # 2. max_w
            ax = axes[0, 1]
            values = [t["max_w"] for t in theta_data]
            ax.plot(range(len(dates)), values, marker='o', linewidth=2, color='orange')
            ax.set_title('Max Position Weight (max_w)')
            ax.set_ylabel('Ratio')
            ax.set_xlabel('Update Count')
            ax.grid(True, alpha=0.3)
            
            # 3. turnover_cap
            ax = axes[1, 0]
            values = [t["turnover_cap"] for t in theta_data]
            ax.plot(range(len(dates)), values, marker='o', linewidth=2, color='green')
            ax.set_title('Turnover Cap')
            ax.set_ylabel('Ratio')
            ax.set_xlabel('Update Count')
            ax.grid(True, alpha=0.3)
            
            # 4. risk_mode
            ax = axes[1, 1]
            risk_mode_map = {"risk_on": 1.0, "neutral": 0.5, "risk_off": 0.0}
            values = [risk_mode_map.get(t["risk_mode"], 0.5) for t in theta_data]
            ax.plot(range(len(dates)), values, marker='o', linewidth=2, color='red')
            ax.set_title('Risk Mode')
            ax.set_ylabel('Mode Value')
            ax.set_ylim(-0.1, 1.1)
            ax.set_yticks([0.0, 0.5, 1.0])
            ax.set_yticklabels(['risk_off', 'neutral', 'risk_on'])
            ax.set_xlabel('Update Count')
            ax.grid(True, alpha=0.3)
            
            # 5. enter_th
            ax = axes[2, 0]
            values = [t["enter_th"] for t in theta_data]
            ax.plot(range(len(dates)), values, marker='o', linewidth=2, color='purple')
            ax.set_title('Enter Threshold (enter_th)')
            ax.set_ylabel('Threshold')
            ax.set_xlabel('Update Count')
            ax.grid(True, alpha=0.3)
            
            # 6. exit_th
            ax = axes[2, 1]
            values = [t["exit_th"] for t in theta_data]
            ax.plot(range(len(dates)), values, marker='o', linewidth=2, color='brown')
            ax.set_title('Exit Threshold (exit_th)')
            ax.set_ylabel('Threshold')
            ax.set_xlabel('Update Count')
            ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            # 保存图表
            chart_file = self.output_dir / 'assets' / 'theta_changes.png'
            chart_file.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(chart_file, dpi=150, bbox_inches='tight')
            plt.close()
            
            logger.info(f"参数变化图表已保存: {chart_file}")
        except Exception as e:
            logger.error(f"生成参数变化图表失败: {e}", exc_info=True)

