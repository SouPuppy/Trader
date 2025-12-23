"""
回测报告生成模块
生成交易报告和走势图
"""
import json
import math
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from jinja2 import Environment, FileSystemLoader, select_autoescape
from trader.logger import get_logger


logger = get_logger(__name__)

# 报告格式常量
REPORT_WIDTH = 80
SECTION_SEPARATOR = "-" * REPORT_WIDTH
MAIN_SEPARATOR = "=" * REPORT_WIDTH


class BacktestReport:
    """回测报告生成器"""
    
    def __init__(self, output_dir: Optional[Path] = None):
        """
        初始化报告生成器
        
        Args:
            output_dir: 输出目录，如果为 None 则使用默认目录
        """
        if output_dir is None:
            from trader.config import PROJECT_ROOT
            output_dir = PROJECT_ROOT / 'output' / 'backtest'
        
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 记录每日账户状态
        self.daily_records: List[Dict] = []
        
        # 初始化 Jinja2 环境
        template_dir = Path(__file__).parent
        self.jinja_env = Environment(
            loader=FileSystemLoader(str(template_dir)),
            autoescape=select_autoescape(['html', 'xml']),
            trim_blocks=True,
            lstrip_blocks=True
        )
        
        # 添加自定义过滤器
        def strftime_filter(value, fmt='%Y-%m-%d'):
            if isinstance(value, datetime):
                return value.strftime(fmt)
            elif isinstance(value, str):
                try:
                    dt = pd.to_datetime(value)
                    if isinstance(dt, pd.Timestamp):
                        return dt.strftime(fmt)
                    return dt.strftime(fmt)
                except:
                    return str(value)
            return str(value)
        
        self.jinja_env.filters['strftime'] = strftime_filter
    
    def record_daily_state(self, date: str, account, market_prices: Dict[str, float]):
        """
        记录每日账户状态
        
        Args:
            date: 日期
            account: 账户实例
            market_prices: 市场价格字典
        """
        equity = account.equity(market_prices)
        profit = account.get_total_profit(market_prices)
        return_pct = account.get_total_return(market_prices)
        
        # 计算持仓市值
        positions_value = 0.0
        positions_detail = {}
        for stock_code, position in account.positions.items():
            if stock_code in market_prices:
                price = market_prices[stock_code]
                value = position["shares"] * price
                positions_value += value
                positions_detail[stock_code] = {
                    "shares": position["shares"],
                    "average_price": position["average_price"],
                    "current_price": price,
                    "market_value": value,
                    "profit": (price - position["average_price"]) * position["shares"]
                }
        
        record = {
            "date": date,
            "cash": account.cash,
            "positions_value": positions_value,
            "equity": equity,
            "profit": profit,
            "return_pct": return_pct,
            "positions": positions_detail.copy()
        }
        
        self.daily_records.append(record)
    
    def generate_report(self, account, stock_code: str, start_date: str, end_date: str):
        """
        生成完整的回测报告
        
        Args:
            account: 账户实例
            stock_code: 股票代码
            start_date: 开始日期
            end_date: 结束日期
        """
        logger.info("生成回测报告...")
        
        # 先生成走势图（需要在 Markdown 中引用）
        chart_file = self._generate_charts(stock_code, start_date, end_date)
        
        # 生成 Markdown 报告（包含图表引用）
        markdown_report = self._generate_markdown_report(
            account, stock_code, start_date, end_date, chart_file
        )
        
        # 保存 Markdown 报告
        report_file = self.output_dir / f"backtest_report_{stock_code}_{start_date}_{end_date}.md"
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(markdown_report)
        logger.info(f"Markdown 报告已保存: {report_file}")
        
        # 生成 JSON 报告
        json_report = self._generate_json_report(account, stock_code, start_date, end_date)
        json_file = self.output_dir / f"backtest_report_{stock_code}_{start_date}_{end_date}.json"
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(json_report, f, indent=2, ensure_ascii=False, default=str)
        logger.info(f"JSON 报告已保存: {json_file}")
        
        return report_file
    
    def _generate_markdown_report(self, account, stock_code: str, start_date: str, end_date: str, chart_file: Optional[Path] = None) -> str:
        """使用模板生成 Markdown 报告"""
        # 准备模板数据
        template_data = {
            'stock_code': stock_code,
            'start_date': start_date,
            'end_date': end_date,
            'trading_days': len(self.daily_records),
            'initial_cash': account.initial_cash
        }
        
        # 添加最终状态
        if self.daily_records:
            final_record = self.daily_records[-1]
            template_data['final_state'] = {
                'cash': final_record['cash'],
                'positions_value': final_record['positions_value'],
                'equity': final_record['equity'],
                'profit': final_record['profit'],
                'return_pct': final_record['return_pct'],
                'positions': final_record['positions']
            }
        else:
            template_data['final_state'] = {
                'cash': account.cash,
                'positions_value': 0.0,
                'equity': account.cash,
                'profit': 0.0,
                'return_pct': 0.0,
                'positions': {}
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
            
            template_data['statistics'] = {
                'max_equity': max(equities),
                'min_equity': min(equities),
                'max_return_pct': max(returns),
                'min_return_pct': min(returns),
                'max_drawdown_pct': max_drawdown
            }
            
            if sharpe_info:
                template_data['statistics']['sharpe_annual'] = sharpe_info['sharpe_annual']
                template_data['statistics']['sharpe_daily'] = sharpe_info['sharpe_daily']
        else:
            template_data['statistics'] = {
                'max_equity': account.initial_cash,
                'min_equity': account.initial_cash,
                'max_return_pct': 0.0,
                'min_return_pct': 0.0,
                'max_drawdown_pct': 0.0
            }
        
        # 添加图表文件路径（相对路径，用于 Markdown 中的图片引用）
        if chart_file and chart_file.exists():
            # 使用相对于 Markdown 文件的路径
            chart_relative_path = chart_file.name
            template_data['chart_file'] = chart_relative_path
        else:
            template_data['chart_file'] = None
        
        # 渲染模板
        template = self.jinja_env.get_template('report_template.md.j2')
        return template.render(**template_data)
    
    def _format_header(self, stock_code: str, start_date: str, end_date: str) -> List[str]:
        """格式化报告头部"""
        return [
            MAIN_SEPARATOR,
            "回测报告".center(REPORT_WIDTH),
            MAIN_SEPARATOR,
            "",
            f"股票代码: {stock_code}",
            f"回测期间: {start_date} 至 {end_date}",
            f"交易日数: {len(self.daily_records)}",
            ""
        ]
    
    def _format_account_summary(self, account) -> List[str]:
        """格式化账户摘要"""
        final_record = self.daily_records[-1]
        lines = [
            "账户摘要",
            SECTION_SEPARATOR,
            f"初始资金:     {account.initial_cash:>15,.2f} 元",
            f"最终现金:     {final_record['cash']:>15,.2f} 元",
            f"最终持仓市值: {final_record['positions_value']:>15,.2f} 元",
            f"最终总权益:   {final_record['equity']:>15,.2f} 元",
            f"总盈亏:       {final_record['profit']:>15+,.2f} 元",
            f"总收益率:     {final_record['return_pct']:>15+.2f}%",
            ""
        ]
        return lines
    
    def _format_trade_statistics(self, trades: List[Dict]) -> List[str]:
        """格式化交易统计信息"""
        buy_trades = [t for t in trades if t['type'] == 'buy']
        sell_trades = [t for t in trades if t['type'] == 'sell']
        
        total_buy_cost = sum(t['cost'] for t in buy_trades)
        total_sell_revenue = sum(t.get('revenue', 0) for t in sell_trades)
        total_sell_profit = sum(t.get('profit', 0) for t in sell_trades)
        
        avg_buy_price = sum(t['price'] * t['shares'] for t in buy_trades) / sum(t['shares'] for t in buy_trades) if buy_trades else 0
        avg_sell_price = sum(t['price'] * t['shares'] for t in sell_trades) / sum(t['shares'] for t in sell_trades) if sell_trades else 0
        
        lines = [
            "交易统计",
            SECTION_SEPARATOR,
            f"总交易次数:       {len(trades):>10} 次",
            f"  买入次数:       {len(buy_trades):>10} 次",
            f"  卖出次数:       {len(sell_trades):>10} 次",
            "",
            f"买入总成本:       {total_buy_cost:>15,.2f} 元",
            f"卖出总收入:       {total_sell_revenue:>15,.2f} 元",
            f"已实现盈亏:       {total_sell_profit:>15+,.2f} 元",
            "",
            f"平均买入价格:     {avg_buy_price:>15,.2f} 元/股" if buy_trades else "",
            f"平均卖出价格:     {avg_sell_price:>15,.2f} 元/股" if sell_trades else "",
            ""
        ]
        return [line for line in lines if line]  # 移除空行
    
    def _format_trades(self, trades: List[Dict]) -> List[str]:
        """格式化交易记录"""
        lines = [
            "交易记录",
            SECTION_SEPARATOR
        ]
        
        if not trades:
            lines.append("(无交易记录)")
            lines.append("")
            return lines
        
        # 按日期排序
        sorted_trades = sorted(trades, key=lambda t: t['date'] if isinstance(t['date'], datetime) else pd.to_datetime(t['date']))
        
        for i, trade in enumerate(sorted_trades, 1):
            date_str = trade['date'].strftime('%Y-%m-%d') if isinstance(trade['date'], datetime) else str(trade['date'])
            
            if trade['type'] == 'buy':
                lines.append(
                    f"{i:>4}. [{date_str}] 买入 {trade['stock_code']:>10} | "
                    f"{trade['shares']:>6} 股 @ {trade['price']:>8.2f} | "
                    f"成本: {trade['cost']:>10,.2f} 元"
                )
            else:
                lines.append(
                    f"{i:>4}. [{date_str}] 卖出 {trade['stock_code']:>10} | "
                    f"{trade['shares']:>6} 股 @ {trade['price']:>8.2f} | "
                    f"收入: {trade['revenue']:>10,.2f} 元 | "
                    f"利润: {trade['profit']:>10+,.2f} 元"
                )
        
        lines.append("")
        return lines
    
    def _format_final_positions(self) -> List[str]:
        """格式化最终持仓"""
        lines = [
            "最终持仓",
            SECTION_SEPARATOR
        ]
        
        if not self.daily_records or not self.daily_records[-1]['positions']:
            lines.append("(无持仓)")
            lines.append("")
            return lines
        
        positions = self.daily_records[-1]['positions']
        for stock_code, pos in positions.items():
            profit_pct = (pos['profit'] / (pos['shares'] * pos['average_price'])) * 100 if pos['shares'] * pos['average_price'] > 0 else 0
            lines.append(
                f"{stock_code:>10} | "
                f"{pos['shares']:>6} 股 | "
                f"成本: {pos['average_price']:>8.2f} | "
                f"现价: {pos['current_price']:>8.2f} | "
                f"市值: {pos['market_value']:>12,.2f} 元 | "
                f"盈亏: {pos['profit']:>10+,.2f} 元 ({profit_pct:+.2f}%)"
            )
        
        lines.append("")
        return lines
    
    def _format_statistics(self) -> List[str]:
        """格式化统计信息"""
        equities = [r['equity'] for r in self.daily_records]
        returns = [r['return_pct'] for r in self.daily_records]
        
        max_drawdown = self._calculate_max_drawdown(equities)
        daily_returns, sharpe_info = self._calculate_daily_returns(risk_free_rate_annual=0.0)
        
        lines = [
            "统计信息",
            SECTION_SEPARATOR,
            f"最高权益:       {max(equities):>15,.2f} 元",
            f"最低权益:       {min(equities):>15,.2f} 元",
            f"最高收益率:     {max(returns):>15+.2f}%",
            f"最低收益率:     {min(returns):>15+.2f}%",
            f"最大回撤:       {max_drawdown:>15.2f}%"
        ]
        
        if sharpe_info:
            lines.extend([
                f"年化夏普比率:   {sharpe_info['sharpe_annual']:>15.4f}",
                f"日频夏普比率:   {sharpe_info['sharpe_daily']:>15.4f}"
            ])
        
        lines.append("")
        return lines
    
    def _generate_json_report(self, account, stock_code: str, start_date: str, end_date: str) -> Dict:
        """生成 JSON 报告"""
        report = {
            "stock_code": stock_code,
            "start_date": start_date,
            "end_date": end_date,
            "trading_days": len(self.daily_records),
            "initial_cash": account.initial_cash
        }
        
        if self.daily_records:
            final_record = self.daily_records[-1]
            report["final_state"] = {
                "cash": final_record['cash'],
                "positions_value": final_record['positions_value'],
                "equity": final_record['equity'],
                "profit": final_record['profit'],
                "return_pct": final_record['return_pct'],
                "positions": final_record['positions']
            }
            
            # 交易统计
            buy_trades = [t for t in account.trades if t['type'] == 'buy']
            sell_trades = [t for t in account.trades if t['type'] == 'sell']
            
            report["trade_statistics"] = {
                "total_trades": len(account.trades),
                "buy_count": len(buy_trades),
                "sell_count": len(sell_trades),
                "total_buy_cost": sum(t['cost'] for t in buy_trades),
                "total_sell_revenue": sum(t.get('revenue', 0) for t in sell_trades),
                "realized_profit": sum(t.get('profit', 0) for t in sell_trades)
            }
            
            # 添加统计信息
            equities = [r['equity'] for r in self.daily_records]
            returns = [r['return_pct'] for r in self.daily_records]
            max_drawdown = self._calculate_max_drawdown(equities)
            
            # 计算夏普比率
            daily_returns, sharpe_info = self._calculate_daily_returns(risk_free_rate_annual=0.0)
            
            report["statistics"] = {
                "max_equity": max(equities),
                "min_equity": min(equities),
                "max_return_pct": max(returns),
                "min_return_pct": min(returns),
                "max_drawdown_pct": max_drawdown
            }
            
            # 添加风险指标
            if sharpe_info:
                report["statistics"].update({
                    "sharpe_annual": sharpe_info['sharpe_annual'],
                    "sharpe_daily": sharpe_info['sharpe_daily'],
                    "mean_daily_return": sharpe_info['mean_excess_return'],
                    "std_daily_return": sharpe_info['std_excess_return']
                })
                
                # 详细的夏普比率信息（可选，用于深度分析）
                report["sharpe_ratio_details"] = {
                    "risk_free_rate_annual": sharpe_info['risk_free_rate_annual'],
                    "risk_free_rate_daily": sharpe_info['risk_free_rate_daily'],
                    "num_trading_days": sharpe_info['num_trading_days'],
                    "daily_returns": daily_returns,
                    "excess_returns": sharpe_info['excess_returns']
                }
        
        # 交易记录和每日记录（放在最后，因为可能很大）
        report["trades"] = account.trades
        report["daily_records"] = self.daily_records
        
        return report
    
    def _calculate_max_drawdown(self, equities: List[float]) -> float:
        """计算最大回撤"""
        if not equities:
            return 0.0
        
        max_dd = 0.0
        peak = equities[0]
        
        for equity in equities:
            if equity > peak:
                peak = equity
            dd = (peak - equity) / peak * 100
            if dd > max_dd:
                max_dd = dd
        
        return max_dd
    
    def _calculate_daily_returns(self, risk_free_rate_annual: float = 0.0) -> Tuple[List[float], Dict]:
        """
        计算日收益率序列和夏普比率
        
        Args:
            risk_free_rate_annual: 年化无风险利率（默认0）
            
        Returns:
            (daily_returns, sharpe_info): 日收益率列表和夏普比率详细信息字典
        """
        if len(self.daily_records) < 2:
            return [], {}
        
        # 提取每日净值（按日期排序）
        df = pd.DataFrame(self.daily_records)
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date')
        equities = df['equity'].tolist()
        
        # 计算日收益率序列: r_t = E_t / E_{t-1} - 1
        daily_returns = []
        for i in range(1, len(equities)):
            if equities[i-1] > 0:
                r_t = equities[i] / equities[i-1] - 1
                daily_returns.append(r_t)
            else:
                daily_returns.append(0.0)
        
        if not daily_returns:
            return [], {}
        
        # 日化无风险利率: r_f = (1 + R_f(ann))^(1/252) - 1
        if risk_free_rate_annual == 0.0:
            risk_free_rate_daily = 0.0
        else:
            risk_free_rate_daily = math.pow(1 + risk_free_rate_annual, 1/252) - 1
        
        # 计算超额收益: x_t = r_t - r_f
        excess_returns = [r - risk_free_rate_daily for r in daily_returns]
        
        # 计算样本均值和标准差
        T = len(excess_returns)
        mean_excess_return = sum(excess_returns) / T
        
        # 样本标准差（无偏估计）: s = sqrt((1/(T-1)) * Σ(x_t - x̄)^2)
        variance = sum((x - mean_excess_return) ** 2 for x in excess_returns) / (T - 1) if T > 1 else 0.0
        std_excess_return = math.sqrt(variance) if variance > 0 else 0.0
        
        # 计算日频夏普比率: Sharpe_daily = x̄ / s
        if std_excess_return > 0:
            sharpe_daily = mean_excess_return / std_excess_return
        else:
            sharpe_daily = 0.0
        
        # 年化夏普比率: Sharpe_ann = sqrt(252) * Sharpe_daily
        sharpe_annual = math.sqrt(252) * sharpe_daily
        
        # 构建详细信息字典
        sharpe_info = {
            'risk_free_rate_annual': risk_free_rate_annual,
            'risk_free_rate_daily': risk_free_rate_daily,
            'daily_returns': daily_returns,
            'excess_returns': excess_returns,
            'mean_excess_return': mean_excess_return,
            'std_excess_return': std_excess_return,
            'sharpe_daily': sharpe_daily,
            'sharpe_annual': sharpe_annual,
            'num_trading_days': T,
            'equities': equities
        }
        
        return daily_returns, sharpe_info
    
    def _generate_charts(self, stock_code: str, start_date: str, end_date: str) -> Optional[Path]:
        """生成走势图"""
        if not self.daily_records:
            logger.warning("没有每日记录，无法生成走势图")
            return None
        
        logger.info("生成走势图...")
        
        # 设置中文字体（避免中文显示问题）
        try:
            plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
            plt.rcParams['axes.unicode_minus'] = False
        except:
            pass  # 如果字体设置失败，使用默认字体
        
        # 准备数据
        df = pd.DataFrame(self.daily_records)
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date')
        
        # 计算统计信息
        equities = df['equity'].tolist()
        returns = df['return_pct'].tolist()
        max_drawdown = self._calculate_max_drawdown(equities)
        
        # 计算夏普比率
        daily_returns, sharpe_info = self._calculate_daily_returns(risk_free_rate_annual=0.0)
        
        # 计算回撤序列
        drawdowns = []
        peak = equities[0]
        for equity in equities:
            if equity > peak:
                peak = equity
            dd = (peak - equity) / peak * 100
            drawdowns.append(dd)
        df['drawdown'] = drawdowns
        
        # 计算滚动夏普比率（30天窗口）
        rolling_sharpe = []
        rolling_sharpe_dates = []
        if sharpe_info and len(sharpe_info['daily_returns']) >= 30:
            daily_returns_list = sharpe_info['daily_returns']
            risk_free_rate_daily = sharpe_info['risk_free_rate_daily']
            window = 30
            
            # 获取日期（从第二天开始，因为日收益率从第二天开始）
            dates_for_returns = df['date'].iloc[1:].tolist()
            
            for i in range(window - 1, len(daily_returns_list)):
                # 获取窗口内的收益率
                window_returns = daily_returns_list[i - window + 1:i + 1]
                # 计算超额收益
                window_excess = [r - risk_free_rate_daily for r in window_returns]
                
                # 计算均值和标准差
                if len(window_excess) > 1:
                    mean_excess = sum(window_excess) / len(window_excess)
                    variance = sum((x - mean_excess) ** 2 for x in window_excess) / (len(window_excess) - 1)
                    std_excess = math.sqrt(variance) if variance > 0 else 0.0
                    
                    # 计算滚动夏普比率
                    if std_excess > 0:
                        rolling_sharpe_val = mean_excess / std_excess
                    else:
                        rolling_sharpe_val = 0.0
                else:
                    rolling_sharpe_val = 0.0
                
                rolling_sharpe.append(rolling_sharpe_val)
                rolling_sharpe_dates.append(dates_for_returns[i])
        
        # 创建图表 - 5个子图：权益、回撤、现金/持仓、收益率、滚动夏普
        fig, axes = plt.subplots(5, 1, figsize=(12, 14))
        # 使用英文标题避免字体问题
        fig.suptitle(f'Backtest Report - {stock_code} ({start_date} to {end_date})', fontsize=14)
        
        # 1. 账户权益曲线
        ax1 = axes[0]
        ax1.plot(df['date'], df['equity'], label='Total Equity', linewidth=2, color='blue')
        ax1.axhline(y=self.daily_records[0]['equity'], color='gray', linestyle='--', 
                   label=f'Initial Capital ({self.daily_records[0]["equity"]:,.0f})', alpha=0.7)
        ax1.set_ylabel('Equity (CNY)', fontsize=10)
        ax1.set_title('Total Equity Trend', fontsize=12)
        ax1.legend(loc='best')
        ax1.grid(True, alpha=0.3)
        ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        ax1.xaxis.set_major_locator(mdates.MonthLocator())
        plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45)
        
        # 2. 回撤曲线
        ax2 = axes[1]
        ax2.fill_between(df['date'], df['drawdown'], 0, alpha=0.3, color='red', label='Drawdown')
        ax2.plot(df['date'], df['drawdown'], linewidth=1.5, color='darkred')
        ax2.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        ax2.set_ylabel('Drawdown (%)', fontsize=10)
        ax2.set_title(f'Drawdown Trend (Max: {max_drawdown:.2f}%)', fontsize=12)
        ax2.legend(loc='best')
        ax2.grid(True, alpha=0.3)
        ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        ax2.xaxis.set_major_locator(mdates.MonthLocator())
        plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45)
        
        # 3. 现金和持仓市值
        ax3 = axes[2]
        ax3.plot(df['date'], df['cash'], label='Cash', linewidth=1.5, color='green')
        ax3.plot(df['date'], df['positions_value'], label='Positions Value', linewidth=1.5, color='orange')
        ax3.set_ylabel('Amount (CNY)', fontsize=10)
        ax3.set_title('Cash vs Positions Value', fontsize=12)
        ax3.legend(loc='best')
        ax3.grid(True, alpha=0.3)
        ax3.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        ax3.xaxis.set_major_locator(mdates.MonthLocator())
        plt.setp(ax3.xaxis.get_majorticklabels(), rotation=45)
        
        # 4. 收益率曲线
        ax4 = axes[3]
        ax4.plot(df['date'], df['return_pct'], label='Return Rate', linewidth=2, color='red')
        ax4.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        ax4.set_ylabel('Return Rate (%)', fontsize=10)
        ax4.set_title('Cumulative Return Trend', fontsize=12)
        ax4.legend(loc='best')
        ax4.grid(True, alpha=0.3)
        ax4.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        ax4.xaxis.set_major_locator(mdates.MonthLocator())
        plt.setp(ax4.xaxis.get_majorticklabels(), rotation=45)
        
        # 5. 滚动夏普比率
        ax5 = axes[4]
        if rolling_sharpe and rolling_sharpe_dates:
            ax5.plot(rolling_sharpe_dates, rolling_sharpe, label='Rolling Sharpe (30d)', linewidth=2, color='purple')
            # 显示整体夏普比率水平线
            if sharpe_info:
                ax5.axhline(y=sharpe_info['sharpe_daily'], color='gray', linestyle='--', 
                           label=f'Overall Sharpe Daily: {sharpe_info["sharpe_daily"]:.4f}', alpha=0.7)
            ax5.axhline(y=0, color='black', linestyle='-', alpha=0.3, linewidth=0.5)
            ax5.set_ylabel('Sharpe Ratio (Daily)', fontsize=10)
            ax5.set_xlabel('Date', fontsize=10)
            ax5.set_title('Rolling Sharpe Ratio (30-day window)', fontsize=12)
            ax5.legend(loc='best')
            ax5.grid(True, alpha=0.3)
            ax5.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
            ax5.xaxis.set_major_locator(mdates.MonthLocator())
            plt.setp(ax5.xaxis.get_majorticklabels(), rotation=45)
        else:
            ax5.text(0.5, 0.5, 'Insufficient data for rolling Sharpe calculation\n(need at least 30 trading days)',
                    transform=ax5.transAxes, ha='center', va='center', fontsize=10)
            ax5.set_ylabel('Sharpe Ratio (Daily)', fontsize=10)
            ax5.set_xlabel('Date', fontsize=10)
            ax5.set_title('Rolling Sharpe Ratio (30-day window)', fontsize=12)
            ax5.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # 保存图表
        chart_file = self.output_dir / f"backtest_charts_{stock_code}_{start_date}_{end_date}.png"
        plt.savefig(chart_file, dpi=150, bbox_inches='tight')
        plt.close()
        
        logger.info(f"走势图已保存: {chart_file}")
        return chart_file

