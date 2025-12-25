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
    
    def __init__(self, output_dir: Optional[Path] = None, title: Optional[str] = None):
        """
        初始化报告生成器
        
        Args:
            output_dir: 输出目录，如果为 None 则使用默认目录
            title: 报告标题，用于创建子文件夹和文件名
        """
        if output_dir is None:
            from trader.config import PROJECT_ROOT
            output_dir = PROJECT_ROOT / 'output' / 'backtest'
        
        base_output_dir = Path(output_dir)
        
        # 如果有 title，创建子文件夹
        if title:
            # 直接使用 title，不做任何修改
            self.output_dir = base_output_dir / title
        else:
            self.output_dir = base_output_dir
        
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.title = title or "Backtest"
        
        # 记录每日账户状态
        self.daily_records: List[Dict] = []
        self.train_test_split_date: Optional[str] = None  # 训练/测试分割日期（用于图表显示）
        
        # 初始化 Jinja2 环境
        template_dir = Path(__file__).parent / 'template'
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
        position_weights = account.get_position_weights(market_prices)
        position_values = account.get_position_values(market_prices)
        position_profits = account.get_position_profits(market_prices)
        position_returns = account.get_position_returns(market_prices)
        
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
                    "profit": (price - position["average_price"]) * position["shares"],
                    "weight": position_weights.get(stock_code, 0.0),
                    "return_pct": position_returns.get(stock_code, 0.0)
                }
        
        # 判断是否在测试期
        is_test_period = False
        if self.train_test_split_date:
            is_test_period = date >= self.train_test_split_date
        
        record = {
            "date": date,
            "cash": account.cash,
            "positions_value": positions_value,
            "equity": equity,
            "profit": profit,
            "return_pct": return_pct,
            "positions": positions_detail.copy(),
            "position_weights": position_weights.copy(),
            "position_values": position_values.copy(),
            "position_profits": position_profits.copy(),
            "position_returns": position_returns.copy(),
            "is_test_period": is_test_period
        }
        
        self.daily_records.append(record)
    
    def set_train_test_split_date(self, split_date: Optional[str]):
        """
        设置训练/测试分割日期
        
        Args:
            split_date: 分割日期（格式: YYYY-MM-DD），如果为 None 则不分割
        """
        self.train_test_split_date = split_date
    
    def generate_report(self, account, stock_code: str, start_date: str, end_date: str, 
                       all_stock_codes: Optional[List[str]] = None, is_multi_asset: bool = False):
        """
        生成完整的回测报告（支持多股票和多资产）
        
        Args:
            account: 账户实例
            stock_code: 主要股票代码（用于文件名）
            start_date: 开始日期
            end_date: 结束日期
            all_stock_codes: 所有股票代码列表（用于多资产回测）
            is_multi_asset: 是否为多资产组合策略
        """
        logger.info("生成回测报告...")
        
        # 如果没有提供所有股票代码，从账户持仓中提取
        if all_stock_codes is None:
            all_stock_codes = list(account.positions.keys())
            # 如果持仓为空，从交易记录中提取
            if not all_stock_codes:
                all_stock_codes = list(set(trade['stock_code'] for trade in account.trades))
            # 如果还是没有，使用主要股票代码
            if not all_stock_codes:
                all_stock_codes = [stock_code]
        
        # 如果是多资产策略，使用多资产报告模板
        if is_multi_asset or len(all_stock_codes) > 1:
            return self.generate_multi_asset_report(
                account, all_stock_codes, start_date, end_date
            )
        
        # 单股票或多股票分别测试，使用原有模板
        # 先生成走势图（需要在 Markdown 中引用）
        chart_file = self._generate_charts(stock_code, start_date, end_date, account=account)
        
        # 生成 Markdown 报告（包含图表引用）
        markdown_report = self._generate_markdown_report(
            account, stock_code, start_date, end_date, chart_file, all_stock_codes
        )
        
        # 保存 Markdown 报告（使用 title 在文件名中）
        report_file = self.output_dir / f"backtest_report_{self.title}_{stock_code}_{start_date}_{end_date}.md"
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(markdown_report)
        logger.info(f"Markdown 报告已保存: {report_file}")
        
        # 生成 JSON 报告
        json_report = self._generate_json_report(account, stock_code, start_date, end_date, all_stock_codes)
        json_file = self.output_dir / f"backtest_report_{self.title}_{stock_code}_{start_date}_{end_date}.json"
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(json_report, f, indent=2, ensure_ascii=False, default=str)
        logger.info(f"JSON 报告已保存: {json_file}")
        
        return report_file
    
    def generate_multi_asset_report(
        self,
        account,
        stock_codes: List[str],
        start_date: str,
        end_date: str,
        strategy_name: Optional[str] = None,
        strategy_params: Optional[Dict] = None
    ) -> Path:
        """
        生成多资产组合报告
        
        Args:
            account: 账户实例
            stock_codes: 所有股票代码列表
            start_date: 开始日期
            end_date: 结束日期
            strategy_name: 策略名称（可选）
            strategy_params: 策略参数字典（可选）
        
        Returns:
            报告文件路径
        """
        logger.info("生成多资产组合报告...")
        
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
        
        # 准备模板数据
        template_data = {
            'title': self.title,
            'strategy_name': strategy_name or self.title,
            'start_date': start_date,
            'end_date': end_date,
            'trading_days': len(self.daily_records),
            'stock_count': len(stock_codes),
            'initial_cash': account.initial_cash,
            'final_state': final_state,
            'stock_performance': stock_performance,
            'strategy_params': strategy_params or {},
            'report_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
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
        if len(self.daily_records) > 1 and account.trades:
            execution_metrics = self._calculate_execution_metrics(account, len(self.daily_records))
            if execution_metrics:
                template_data['execution_metrics'] = execution_metrics
        
        # 生成走势图
        chart_file = self._generate_charts(stock_codes[0] if stock_codes else "portfolio", start_date, end_date, all_stock_codes=stock_codes, account=account)
        template_data['chart_file'] = chart_file.name if chart_file else None
        
        # 渲染模板
        template = self.jinja_env.get_template('multi_asset_report_template.md.j2')
        markdown_report = template.render(**template_data)
        
        # 保存报告
        report_file = self.output_dir / f"multi_asset_report_{self.title}_{start_date}_{end_date}.md"
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(markdown_report)
        
        logger.info(f"多资产组合报告已保存: {report_file}")
        return report_file
    
    def _generate_markdown_report(self, account, stock_code: str, start_date: str, end_date: str, 
                                 chart_file: Optional[Path] = None, all_stock_codes: Optional[List[str]] = None) -> str:
        """使用模板生成 Markdown 报告"""
        # 准备模板数据
        template_data = {
            'title': self.title,
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
                tail_risk = self._calculate_tail_risk(daily_returns)
                template_data['statistics'].update(tail_risk)
        else:
            template_data['statistics'] = {
                'max_equity': account.initial_cash,
                'min_equity': account.initial_cash,
                'max_return_pct': 0.0,
                'min_return_pct': 0.0,
                'max_drawdown_pct': 0.0,
                'cagr': 0.0,
                'calmar': 0.0
            }
        
        # 添加交易指标
        trade_metrics = self._calculate_trade_metrics(account)
        if trade_metrics:
            template_data['trade_metrics'] = trade_metrics
        
        # 添加执行指标
        execution_metrics = self._calculate_execution_metrics(account, len(self.daily_records))
        if execution_metrics:
            template_data['execution_metrics'] = execution_metrics
        
        # 添加图表文件路径（相对路径，用于 Markdown 中的图片引用）
        if chart_file and chart_file.exists():
            # 使用相对于 Markdown 文件的路径
            chart_relative_path = chart_file.name
            template_data['chart_file'] = chart_relative_path
        else:
            template_data['chart_file'] = None
        
        # 渲染模板
        template = self.jinja_env.get_template('single_stock_report_template.md.j2')
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
    
    def _generate_json_report(self, account, stock_code: str, start_date: str, end_date: str, 
                              all_stock_codes: Optional[List[str]] = None) -> Dict:
        """生成 JSON 报告（支持多股票）"""
        if all_stock_codes is None:
            all_stock_codes = [stock_code]
        
        report = {
            "stock_code": stock_code,
            "all_stock_codes": all_stock_codes,
            "is_multi_asset": len(all_stock_codes) > 1,
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
            
            # 添加 CAGR
            start_equity = equities[0]
            end_equity = equities[-1]
            cagr = self._calculate_cagr(start_equity, end_equity, len(self.daily_records))
            report["statistics"]["cagr"] = cagr
            
            # 添加 Calmar 比率
            calmar = self._calculate_calmar(cagr, max_drawdown)
            report["statistics"]["calmar"] = calmar
            
            # 添加波动率
            if daily_returns:
                volatility = self._calculate_volatility(daily_returns)
                report["statistics"]["volatility_annual"] = volatility
            
            # 添加 Sortino 比率
            if daily_returns:
                sortino_info = self._calculate_sortino(daily_returns, risk_free_rate_annual=0.0)
                if sortino_info:
                    report["statistics"]["sortino_annual"] = sortino_info.get('sortino_annual', 0.0)
                    report["statistics"]["sortino_daily"] = sortino_info.get('sortino_daily', 0.0)
            
            # 添加尾部风险
            if daily_returns:
                tail_risk = self._calculate_tail_risk(daily_returns)
                report["statistics"].update(tail_risk)
            
            # 添加交易指标
            trade_metrics = self._calculate_trade_metrics(account)
            if trade_metrics:
                report["trade_metrics"] = trade_metrics
            
            # 添加执行指标
            execution_metrics = self._calculate_execution_metrics(account, len(self.daily_records))
            if execution_metrics:
                report["execution_metrics"] = execution_metrics
        
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
    
    def _calculate_cagr(self, start_equity: float, end_equity: float, trading_days: int) -> float:
        """
        计算年化收益率 (CAGR)
        
        Args:
            start_equity: 初始权益
            end_equity: 最终权益
            trading_days: 交易日数
            
        Returns:
            float: CAGR (百分比)
        """
        if start_equity <= 0 or trading_days <= 0:
            return 0.0
        
        # CAGR = (End/Start)^(252/TradingDays) - 1
        years = trading_days / 252.0
        if years <= 0:
            return 0.0
        
        cagr = (math.pow(end_equity / start_equity, 1.0 / years) - 1.0) * 100
        return cagr
    
    def _calculate_volatility(self, daily_returns: List[float]) -> float:
        """
        计算年化波动率
        
        Args:
            daily_returns: 日收益率序列
            
        Returns:
            float: 年化波动率 (百分比)
        """
        if len(daily_returns) < 2:
            return 0.0
        
        # 计算日收益率的标准差
        mean_return = sum(daily_returns) / len(daily_returns)
        variance = sum((r - mean_return) ** 2 for r in daily_returns) / (len(daily_returns) - 1)
        std_daily = math.sqrt(variance) if variance > 0 else 0.0
        
        # 年化波动率 = 日波动率 * sqrt(252)
        volatility_annual = std_daily * math.sqrt(252) * 100
        return volatility_annual
    
    def _calculate_sortino(self, daily_returns: List[float], risk_free_rate_annual: float = 0.0) -> Dict:
        """
        计算 Sortino 比率（只惩罚下行波动）
        
        Args:
            daily_returns: 日收益率序列
            risk_free_rate_annual: 年化无风险利率
            
        Returns:
            Dict: Sortino 比率信息
        """
        if len(daily_returns) < 2:
            return {}
        
        # 日化无风险利率
        if risk_free_rate_annual == 0.0:
            risk_free_rate_daily = 0.0
        else:
            risk_free_rate_daily = math.pow(1 + risk_free_rate_annual, 1/252) - 1
        
        # 计算超额收益
        excess_returns = [r - risk_free_rate_daily for r in daily_returns]
        mean_excess_return = sum(excess_returns) / len(excess_returns)
        
        # 只计算下行波动（负的超额收益）
        downside_returns = [r for r in excess_returns if r < 0]
        if len(downside_returns) < 2:
            return {
                'sortino_daily': 0.0,
                'sortino_annual': 0.0,
                'downside_std': 0.0
            }
        
        # 下行标准差
        downside_variance = sum(r ** 2 for r in downside_returns) / (len(downside_returns) - 1)
        downside_std = math.sqrt(downside_variance) if downside_variance > 0 else 0.0
        
        # Sortino 比率
        if downside_std > 0:
            sortino_daily = mean_excess_return / downside_std
        else:
            sortino_daily = 0.0
        
        # 年化 Sortino 比率
        sortino_annual = math.sqrt(252) * sortino_daily
        
        return {
            'sortino_daily': sortino_daily,
            'sortino_annual': sortino_annual,
            'downside_std': downside_std,
            'mean_excess_return': mean_excess_return
        }
    
    def _calculate_calmar(self, cagr: float, max_drawdown: float):
        """
        计算 Calmar 比率 (CAGR / Max Drawdown)
        
        Args:
            cagr: 年化收益率 (百分比)
            max_drawdown: 最大回撤 (百分比)
            
        Returns:
            float 或 None: Calmar 比率（None 表示无穷大）
        """
        if max_drawdown == 0:
            # 使用 None 表示无穷大，而不是 float('inf')，因为 Jinja2 模板不支持 float('inf')
            return None if cagr != 0 else 0.0
        return cagr / max_drawdown
    
    def _calculate_tail_risk(self, daily_returns: List[float]) -> Dict:
        """
        计算尾部风险指标 (VaR 和 CVaR)
        
        Args:
            daily_returns: 日收益率序列
            
        Returns:
            Dict: 包含 VaR 和 CVaR 的字典
        """
        if not daily_returns:
            return {}
        
        sorted_returns = sorted(daily_returns)
        n = len(sorted_returns)
        
        # VaR 95% 和 99%
        var_95_idx = int(n * 0.05)
        var_99_idx = int(n * 0.01)
        
        var_95 = sorted_returns[var_95_idx] if var_95_idx < n else sorted_returns[0]
        var_99 = sorted_returns[var_99_idx] if var_99_idx < n else sorted_returns[0]
        
        # CVaR (Conditional VaR) = 尾部损失的期望值
        cvar_95 = sum(sorted_returns[:var_95_idx+1]) / (var_95_idx + 1) if var_95_idx >= 0 else 0.0
        cvar_99 = sum(sorted_returns[:var_99_idx+1]) / (var_99_idx + 1) if var_99_idx >= 0 else 0.0
        
        # 极端日跌幅分位数（最差的5%和1%）
        extreme_loss_5pct_idx = max(0, int(n * 0.05) - 1)
        extreme_loss_1pct_idx = max(0, int(n * 0.01) - 1)
        extreme_loss_5pct = sorted_returns[extreme_loss_5pct_idx] if n > 0 else 0.0
        extreme_loss_1pct = sorted_returns[extreme_loss_1pct_idx] if n > 0 else 0.0
        
        return {
            'var_95': var_95 * 100,  # 转换为百分比
            'var_99': var_99 * 100,
            'cvar_95': cvar_95 * 100,
            'cvar_99': cvar_99 * 100,
            'extreme_loss_5pct': extreme_loss_5pct * 100,
            'extreme_loss_1pct': extreme_loss_1pct * 100
        }
    
    def _calculate_trade_metrics(self, account) -> Dict:
        """
        计算交易相关指标
        
        Args:
            account: 账户实例
            
        Returns:
            Dict: 交易指标
        """
        if not account.trades:
            return {}
        
        sell_trades = [t for t in account.trades if t['type'] == 'sell']
        buy_trades = [t for t in account.trades if t['type'] == 'buy']
        
        # Hit Rate (胜率)
        profitable_trades = [t for t in sell_trades if t.get('profit', 0) > 0]
        hit_rate = len(profitable_trades) / len(sell_trades) * 100 if sell_trades else 0.0
        
        # Profit Factor (总盈利 / 总亏损)
        total_profit = sum(t.get('profit', 0) for t in sell_trades if t.get('profit', 0) > 0)
        total_loss = abs(sum(t.get('profit', 0) for t in sell_trades if t.get('profit', 0) < 0))
        # 使用 None 表示无穷大，而不是 float('inf')，因为 Jinja2 模板不支持 float('inf')
        profit_factor = total_profit / total_loss if total_loss > 0 else (None if total_profit > 0 else 0.0)
        
        # Average Trade Return (单笔平均收益)
        avg_trade_return = sum(t.get('profit', 0) for t in sell_trades) / len(sell_trades) if sell_trades else 0.0
        
        # 计算持仓周期（需要分析买入和卖出配对）
        holding_periods = []
        position_open_dates = {}  # {stock_code: [(date, shares)]}
        
        for trade in sorted(account.trades, key=lambda t: t['date'] if isinstance(t['date'], datetime) else pd.to_datetime(t['date'])):
            stock_code = trade['stock_code']
            date = trade['date'] if isinstance(trade['date'], datetime) else pd.to_datetime(trade['date'])
            
            if trade['type'] == 'buy':
                if stock_code not in position_open_dates:
                    position_open_dates[stock_code] = []
                position_open_dates[stock_code].append((date, trade['shares']))
            else:  # sell
                if stock_code in position_open_dates and position_open_dates[stock_code]:
                    # FIFO 配对
                    remaining_shares = trade['shares']
                    while remaining_shares > 0 and position_open_dates[stock_code]:
                        open_date, open_shares = position_open_dates[stock_code][0]
                        if open_shares <= remaining_shares:
                            holding_periods.append((date - open_date).days)
                            remaining_shares -= open_shares
                            position_open_dates[stock_code].pop(0)
                        else:
                            holding_periods.append((date - open_date).days)
                            position_open_dates[stock_code][0] = (open_date, open_shares - remaining_shares)
                            remaining_shares = 0
        
        avg_holding_period = sum(holding_periods) / len(holding_periods) if holding_periods else 0.0
        
        return {
            'hit_rate': hit_rate,
            'profit_factor': profit_factor,
            'avg_trade_return': avg_trade_return,
            'total_profit': total_profit,
            'total_loss': total_loss,
            'profitable_trades': len(profitable_trades),
            'losing_trades': len(sell_trades) - len(profitable_trades),
            'avg_holding_period_days': avg_holding_period
        }
    
    def _calculate_execution_metrics(self, account, trading_days: int) -> Dict:
        """
        计算执行相关指标
        
        Args:
            account: 账户实例
            trading_days: 交易日数
            
        Returns:
            Dict: 执行指标
        """
        if not account.trades or trading_days == 0:
            return {}
        
        # Turnover (换手率) = 总交易金额 / 平均权益
        total_trade_amount = sum(t['cost'] for t in account.trades if t['type'] == 'buy')
        total_trade_amount += sum(t.get('revenue', 0) for t in account.trades if t['type'] == 'sell')
        
        # 计算平均权益
        if self.daily_records:
            avg_equity = sum(r['equity'] for r in self.daily_records) / len(self.daily_records)
            turnover = (total_trade_amount / 2) / avg_equity if avg_equity > 0 else 0.0  # 除以2因为买入和卖出都计算了
        else:
            turnover = 0.0
        
        # Trading Frequency (交易频率)
        total_trades = len(account.trades)
        trades_per_day = total_trades / trading_days if trading_days > 0 else 0.0
        trades_per_week = trades_per_day * 5  # 假设一周5个交易日
        
        return {
            'turnover': turnover * 100,  # 转换为百分比
            'trades_per_day': trades_per_day,
            'trades_per_week': trades_per_week,
            'total_trade_amount': total_trade_amount
        }
    
    def _calculate_alpha_beta(self, daily_returns: List[float], benchmark_returns: Optional[List[float]] = None) -> Dict:
        """
        计算 Alpha 和 Beta（相对基准）
        
        Args:
            daily_returns: 策略日收益率序列
            benchmark_returns: 基准日收益率序列（如果为 None，则无法计算）
            
        Returns:
            Dict: Alpha 和 Beta 信息
        """
        if benchmark_returns is None or len(daily_returns) != len(benchmark_returns) or len(daily_returns) < 2:
            return {}
        
        # 计算协方差和方差
        mean_strategy = sum(daily_returns) / len(daily_returns)
        mean_benchmark = sum(benchmark_returns) / len(benchmark_returns)
        
        covariance = sum((daily_returns[i] - mean_strategy) * (benchmark_returns[i] - mean_benchmark) 
                          for i in range(len(daily_returns))) / (len(daily_returns) - 1)
        
        benchmark_variance = sum((r - mean_benchmark) ** 2 for r in benchmark_returns) / (len(benchmark_returns) - 1)
        
        # Beta = Cov(strategy, benchmark) / Var(benchmark)
        beta = covariance / benchmark_variance if benchmark_variance > 0 else 0.0
        
        # Alpha = mean_strategy - beta * mean_benchmark (日频)
        alpha_daily = mean_strategy - beta * mean_benchmark
        
        # 年化 Alpha 和 Beta
        alpha_annual = alpha_daily * 252 * 100  # 转换为百分比
        
        return {
            'alpha_daily': alpha_daily,
            'alpha_annual': alpha_annual,
            'beta': beta
        }
    
    def _generate_charts(self, stock_code: str, start_date: str, end_date: str, all_stock_codes: Optional[List[str]] = None, account=None) -> Optional[Path]:
        """生成走势图（支持多股票）"""
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
        
        # 分离训练期和测试期（用于标注，不用于截断图表）
        has_train_test_split = False
        split_date_ts: Optional[pd.Timestamp] = None
        if 'is_test_period' in df.columns and self.train_test_split_date:
            train_df = df[~df['is_test_period']].copy()
            test_df = df[df['is_test_period']].copy()
            has_train_test_split = (not train_df.empty) and (not test_df.empty)
            if has_train_test_split:
                split_date_ts = pd.to_datetime(self.train_test_split_date).normalize()
        else:
            train_df = pd.DataFrame()
            test_df = pd.DataFrame()
        
        # 记录首笔交易日期（用于标注/对齐展示起点）
        first_trade_date_ts: Optional[pd.Timestamp] = None
        if account and getattr(account, "trades", None):
            trade_dates: List[pd.Timestamp] = []
            for trade in account.trades:
                trade_date = trade.get('date')
                if not trade_date:
                    continue
                if isinstance(trade_date, str):
                    trade_dates.append(pd.to_datetime(trade_date).normalize())
                elif isinstance(trade_date, datetime):
                    trade_dates.append(pd.to_datetime(trade_date).normalize())
                elif isinstance(trade_date, pd.Timestamp):
                    trade_dates.append(trade_date.normalize())
            if trade_dates:
                first_trade_date_ts = min(trade_dates)
                logger.info(f"第一笔交易日期: {first_trade_date_ts.strftime('%Y-%m-%d')}")
        
        # 图表展示区间：默认从“首笔交易日”开始（更符合直觉：从真正开始交易那天起画）
        # 若没有交易，则展示完整回测区间
        display_df = df.copy()
        if first_trade_date_ts is not None:
            filtered = df[df["date"] >= first_trade_date_ts].copy()
            if not filtered.empty:
                display_df = filtered
        
        starting_equity = (
            display_df.iloc[0]["equity"]
            if not display_df.empty
            else self.daily_records[0]["equity"]
        )
        
        # 检测是否有多个股票（如果未提供，则从记录中检测）
        if all_stock_codes is None:
            all_stock_codes = set()
            for record in display_df.to_dict('records'):
                if 'positions' in record and record['positions']:
                    all_stock_codes.update(record['positions'].keys())
        else:
            # 如果提供了，转换为 set
            all_stock_codes = set(all_stock_codes)
        
        is_multi_asset = len(all_stock_codes) > 1
        
        # 计算统计信息（基于显示的数据）
        equities = display_df['equity'].tolist()
        returns = display_df['return_pct'].tolist()
        max_drawdown = self._calculate_max_drawdown(equities)
        
        # 计算夏普比率（基于展示数据）
        # 使用 display_df（默认从第一笔交易开始）来计算日收益率
        if len(display_df) > 1:
            # 使用过滤后的 display_df 计算日收益率
            display_equities = display_df['equity'].tolist()
            display_daily_returns = []
            for i in range(1, len(display_equities)):
                if display_equities[i-1] > 0:
                    ret = (display_equities[i] / display_equities[i-1]) - 1.0
                    display_daily_returns.append(ret)
            
            # 计算夏普比率
            if display_daily_returns:
                mean_return = sum(display_daily_returns) / len(display_daily_returns)
                variance = sum((r - mean_return) ** 2 for r in display_daily_returns) / (len(display_daily_returns) - 1) if len(display_daily_returns) > 1 else 0.0
                std_return = math.sqrt(variance) if variance > 0 else 0.0
                sharpe_daily = (mean_return / std_return) if std_return > 0 else 0.0
                sharpe_info = {
                    'daily_returns': display_daily_returns,
                    'sharpe_daily': sharpe_daily,
                    'risk_free_rate_daily': 0.0
                }
            else:
                sharpe_info = None
            daily_returns = display_daily_returns
        else:
            # 如果数据不足，使用原有方法计算
            daily_returns, sharpe_info = self._calculate_daily_returns(risk_free_rate_annual=0.0)
        
        # 计算回撤序列（基于显示的数据）
        drawdowns = []
        peak = equities[0]
        for equity in equities:
            if equity > peak:
                peak = equity
            dd = (peak - equity) / peak * 100
            drawdowns.append(dd)
        display_df['drawdown'] = drawdowns
        
        # 计算滚动夏普比率（30天窗口，基于显示的数据）
        rolling_sharpe = []
        rolling_sharpe_dates = []
        if sharpe_info and len(sharpe_info['daily_returns']) >= 30:
            daily_returns_list = sharpe_info['daily_returns']
            risk_free_rate_daily = sharpe_info['risk_free_rate_daily']
            window = 30
            
            # 获取日期（从第二天开始，因为日收益率从第二天开始）
            dates_for_returns = display_df['date'].iloc[1:].tolist()
            
            # 确保日期和收益率数量匹配
            min_len = min(len(daily_returns_list), len(dates_for_returns))
            daily_returns_list = daily_returns_list[:min_len]
            dates_for_returns = dates_for_returns[:min_len]
            
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
        # 图表标题：回测区间 +（可选）train/test split +（可选）展示起点（首笔交易日）
        title_lines = [f"{self.title} - {stock_code} ({start_date} to {end_date})"]
        if has_train_test_split and split_date_ts is not None:
            title_lines.append(f'Train/Test split: {split_date_ts.strftime("%Y-%m-%d")}')
        if first_trade_date_ts is not None:
            title_lines.append(f'Displayed from first trade: {first_trade_date_ts.strftime("%Y-%m-%d")}')
        fig.suptitle("\n".join(title_lines), fontsize=14)
        
        # 1. 账户权益曲线（展示数据）
        ax1 = axes[0]
        ax1.plot(display_df['date'], display_df['equity'], label='Total Equity', 
                linewidth=2, color='blue')
        
        ax1.axhline(y=starting_equity, color='gray', linestyle='--', 
                   label=f'Starting Equity ({starting_equity:,.0f})', alpha=0.7)
        ax1.set_ylabel('Equity (CNY)', fontsize=10)
        ax1.set_title('Total Equity Trend', fontsize=12)
        ax1.legend(loc='best')
        ax1.grid(True, alpha=0.3)
        ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        ax1.xaxis.set_major_locator(mdates.MonthLocator())
        plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45)
        
        # 2. 回撤曲线（展示数据）
        ax2 = axes[1]
        ax2.fill_between(display_df['date'], display_df['drawdown'], 0, alpha=0.3, color='red', label='Drawdown')
        ax2.plot(display_df['date'], display_df['drawdown'], linewidth=1.5, color='darkred')
        ax2.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        ax2.set_ylabel('Drawdown (%)', fontsize=10)
        ax2.set_title(f'Drawdown Trend (Max: {max_drawdown:.2f}%)', fontsize=12)
        ax2.legend(loc='best')
        ax2.grid(True, alpha=0.3)
        ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        ax2.xaxis.set_major_locator(mdates.MonthLocator())
        plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45)
        
        # 3. 现金和持仓市值（展示数据）
        ax3 = axes[2]
        ax3.plot(display_df['date'], display_df['cash'], label='Cash', linewidth=1.5, color='green')
        ax3.plot(display_df['date'], display_df['positions_value'], label='Total Positions Value', 
                linewidth=2, color='orange', linestyle='-')
        
        # 添加每支股票的持仓市值曲线（如果有多个股票）
        # 从 positions 字段中提取每支股票的市值
        if 'positions' in display_df.columns:
            # 提取所有股票代码
            all_stock_codes = set()
            for positions in display_df['positions']:
                if isinstance(positions, dict):
                    all_stock_codes.update(positions.keys())
            
            if len(all_stock_codes) > 1:
                # 为每支股票绘制持仓市值曲线
                colors = plt.cm.tab10(range(len(all_stock_codes)))
                stock_colors = {code: colors[i] for i, code in enumerate(sorted(all_stock_codes))}
                
                for stock_code in sorted(all_stock_codes):
                    stock_values = []
                    for positions in display_df['positions']:
                        if isinstance(positions, dict) and stock_code in positions:
                            stock_values.append(positions[stock_code].get('market_value', 0.0))
                        else:
                            stock_values.append(0.0)
                    
                    if any(v > 0 for v in stock_values):  # 只绘制有持仓的股票
                        ax3.plot(display_df['date'], stock_values, 
                               label=f'{stock_code}', linewidth=1.5, 
                               color=stock_colors[stock_code], alpha=0.7, linestyle='--')
        
        ax3.set_ylabel('Amount (CNY)', fontsize=10)
        ax3.set_title('Cash vs Positions Value (by Stock)', fontsize=12)
        ax3.legend(loc='best', fontsize=8, ncol=2)
        ax3.grid(True, alpha=0.3)
        ax3.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        ax3.xaxis.set_major_locator(mdates.MonthLocator())
        plt.setp(ax3.xaxis.get_majorticklabels(), rotation=45)
        
        # 4. 收益率曲线（展示数据）
        ax4 = axes[3]
        ax4.plot(display_df['date'], display_df['return_pct'], label='Return Rate', 
                linewidth=2, color='red')
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
        
        # 关键日期标注：train/test split 与首笔交易日（仅标注，不截断）
        try:
            if has_train_test_split and split_date_ts is not None:
                for ax in axes:
                    ax.axvline(split_date_ts, color='black', linestyle='--', alpha=0.5, linewidth=1)
            if first_trade_date_ts is not None:
                for ax in axes:
                    ax.axvline(first_trade_date_ts, color='teal', linestyle=':', alpha=0.5, linewidth=1)
        except Exception:
            # 标注失败不影响图表主体
            pass
        
        plt.tight_layout()
        
        # 保存图表（使用 title 在文件名中）
        chart_title = self.title or "backtest"
        chart_file = self.output_dir / f"backtest_charts_{chart_title}_{stock_code}_{start_date}_{end_date}.png"
        plt.savefig(chart_file, dpi=150, bbox_inches='tight')
        plt.close()
        
        # 如果是多资产回测，生成额外的多股票图表
        if is_multi_asset and all_stock_codes:
            multi_asset_chart_file = self._generate_multi_asset_charts(
                display_df, all_stock_codes, start_date, end_date, self.title or "backtest", account=account
            )
            if multi_asset_chart_file:
                logger.info(f"多资产图表已保存: {multi_asset_chart_file}")
        
        logger.info(f"走势图已保存: {chart_file}")
        return chart_file
    
    def _generate_multi_asset_charts(self, display_df: pd.DataFrame, stock_codes: set, 
                                     start_date: str, end_date: str, title: str, account=None) -> Optional[Path]:
        """
        生成多资产回测的额外图表
        
        Args:
            display_df: 显示数据 DataFrame（已经过滤到从第一笔交易开始）
            stock_codes: 股票代码集合
            start_date: 开始日期
            end_date: 结束日期
            title: 标题字符串
            account: 账户实例（可选，用于确定第一笔交易日期）
            
        Returns:
            Optional[Path]: 图表文件路径
        """
        try:
            stock_codes = sorted(list(stock_codes))
            num_stocks = len(stock_codes)
            
            if num_stocks == 0:
                return None
            
            logger.info(f"生成多资产图表，包含 {num_stocks} 支股票")
            
            # display_df 在 _generate_charts 中已确定展示区间（默认完整回测区间），这里直接使用
            # 准备数据：提取每支股票的权重、市值、收益等
            dates = display_df['date'].tolist()
            
            # 提取每支股票的权重序列
            stock_weights = {code: [] for code in stock_codes}
            stock_values = {code: [] for code in stock_codes}
            stock_returns = {code: [] for code in stock_codes}
            stock_cumulative_returns = {code: [] for code in stock_codes}
            
            for _, row in display_df.iterrows():
                positions = row.get('positions', {})
                position_weights = row.get('position_weights', {})
                position_values = row.get('position_values', {})
                position_returns = row.get('position_returns', {})
                
                for code in stock_codes:
                    stock_weights[code].append(position_weights.get(code, 0.0))
                    stock_values[code].append(position_values.get(code, 0.0))
                    stock_returns[code].append(position_returns.get(code, 0.0))
            
            # 计算累计收益（相对于初始值）
            for code in stock_codes:
                if stock_values[code]:
                    initial_value = stock_values[code][0] if stock_values[code][0] > 0 else 1.0
                    cumulative = []
                    cum_sum = 0.0
                    for i, val in enumerate(stock_values[code]):
                        if i == 0:
                            cumulative.append(0.0)
                        else:
                            prev_val = stock_values[code][i-1] if stock_values[code][i-1] > 0 else initial_value
                            if prev_val > 0:
                                cum_sum += (val - prev_val) / prev_val * 100
                            cumulative.append(cum_sum)
                    stock_cumulative_returns[code] = cumulative
            
            # 创建多资产图表：4个子图
            fig, axes = plt.subplots(2, 2, figsize=(16, 12))
            fig.suptitle(f'{self.title} - Multi-Asset Analysis ({num_stocks} stocks)', fontsize=16)
            
            # 1. 持仓权重变化图（堆叠面积图）
            ax1 = axes[0, 0]
            colors = plt.cm.tab10(range(num_stocks))
            ax1.stackplot(dates, *[stock_weights[code] for code in stock_codes],
                          labels=stock_codes, alpha=0.7, colors=colors)
            ax1.set_ylabel('Position Weight', fontsize=10)
            ax1.set_title('Position Weights Over Time', fontsize=12)
            ax1.legend(loc='upper left', fontsize=8)
            ax1.grid(True, alpha=0.3)
            ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
            ax1.xaxis.set_major_locator(mdates.MonthLocator())
            plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45)
            
            # 2. 持仓市值变化图
            ax2 = axes[0, 1]
            for i, code in enumerate(stock_codes):
                ax2.plot(dates, stock_values[code], label=code, linewidth=2, color=colors[i])
            ax2.set_ylabel('Market Value (CNY)', fontsize=10)
            ax2.set_title('Position Values Over Time', fontsize=12)
            ax2.legend(loc='best', fontsize=8)
            ax2.grid(True, alpha=0.3)
            ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
            ax2.xaxis.set_major_locator(mdates.MonthLocator())
            plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45)
            
            # 3. 累计收益对比图
            ax3 = axes[1, 0]
            for i, code in enumerate(stock_codes):
                if stock_cumulative_returns[code]:
                    ax3.plot(dates, stock_cumulative_returns[code], label=code, 
                            linewidth=2, color=colors[i])
            ax3.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
            ax3.set_ylabel('Cumulative Return (%)', fontsize=10)
            ax3.set_xlabel('Date', fontsize=10)
            ax3.set_title('Cumulative Returns Comparison', fontsize=12)
            ax3.legend(loc='best', fontsize=8)
            ax3.grid(True, alpha=0.3)
            ax3.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
            ax3.xaxis.set_major_locator(mdates.MonthLocator())
            plt.setp(ax3.xaxis.get_majorticklabels(), rotation=45)
            
            # 4. 最终持仓分布饼图（使用最后一天的持仓）
            ax4 = axes[1, 1]
            if display_df.iloc[-1].get('positions'):
                final_positions = display_df.iloc[-1]['positions']
                final_weights = display_df.iloc[-1].get('position_weights', {})
                
                # 准备饼图数据
                pie_labels = []
                pie_sizes = []
                pie_colors_list = []
                
                for i, code in enumerate(stock_codes):
                    if code in final_weights and final_weights[code] > 0:
                        pie_labels.append(code)
                        pie_sizes.append(final_weights[code] * 100)  # 转换为百分比
                        pie_colors_list.append(colors[i])
                
                if pie_sizes:
                    ax4.pie(pie_sizes, labels=pie_labels, colors=pie_colors_list, 
                           autopct='%1.1f%%', startangle=90)
                    ax4.set_title('Final Position Distribution', fontsize=12)
                else:
                    ax4.text(0.5, 0.5, 'No positions at end', 
                            transform=ax4.transAxes, ha='center', va='center', fontsize=10)
                    ax4.set_title('Final Position Distribution', fontsize=12)
            else:
                ax4.text(0.5, 0.5, 'No position data', 
                        transform=ax4.transAxes, ha='center', va='center', fontsize=10)
                ax4.set_title('Final Position Distribution', fontsize=12)
            
            plt.tight_layout()
            
            # 保存多资产图表
            chart_file = self.output_dir / f"multi_asset_charts_{title}_{start_date}_{end_date}.png"
            plt.savefig(chart_file, dpi=150, bbox_inches='tight')
            plt.close()
            
            return chart_file
            
        except Exception as e:
            logger.error(f"生成多资产图表时出错: {e}", exc_info=True)
            return None
    
    def generate_multi_stock_report(
        self,
        results: List[Dict],
        strategy_name: str,
        start_date: str,
        end_date: str,
        initial_cash_per_stock: float,
        strategy_params: Optional[Dict] = None
    ) -> Path:
        """
        生成多股票综合报告
        
        Args:
            results: 回测结果列表，每个结果包含 stock_code, initial_cash, final_equity, profit, return_pct, num_trades 等
            strategy_name: 策略名称
            start_date: 开始日期
            end_date: 结束日期
            initial_cash_per_stock: 每只股票的初始资金
            strategy_params: 策略参数字典（可选）
        
        Returns:
            报告文件路径
        """
        logger.info("生成多股票综合报告...")
        
        # 分离成功和失败的结果
        successful_results = [r for r in results if 'error' not in r]
        failed_results = [r for r in results if 'error' in r]
        
        # 计算统计信息
        total_count = len(results)
        successful_count = len(successful_results)
        total_initial_cash = sum(r['initial_cash'] for r in successful_results)
        total_final_equity = sum(r['final_equity'] for r in successful_results)
        total_profit = sum(r['profit'] for r in successful_results)
        avg_return = sum(r['return_pct'] for r in successful_results) / successful_count if successful_count > 0 else 0.0
        
        # 兼容不同的字段名（num_trades 或 dca_executions 等）
        total_trades = sum(
            r.get('num_trades', r.get('dca_executions', r.get('trades', 0)))
            for r in successful_results
        )
        
        # 创建 assets 文件夹
        assets_dir = self.output_dir / 'assets'
        assets_dir.mkdir(exist_ok=True)
        
        # 统一字段名：为每个结果添加 num_trades 字段（如果不存在）
        # 同时为每个股票生成图表和详细指标
        normalized_results = []
        stock_details = []  # 存储每个股票的详细信息（包含图表和指标）
        
        for r in successful_results:
            normalized_r = r.copy()
            # 统一交易次数字段名
            if 'num_trades' not in normalized_r:
                normalized_r['num_trades'] = normalized_r.get('dca_executions', normalized_r.get('trades', 0))
            
            # 为每个股票生成图表和详细指标
            stock_detail = self._generate_stock_detail(
                r, assets_dir, start_date, end_date
            )
            if stock_detail:
                stock_detail['stock_code'] = r['stock_code']
                stock_details.append(stock_detail)
            
            normalized_results.append(normalized_r)
        
        # 按收益率排序
        top_returns = sorted(normalized_results, key=lambda x: x['return_pct'], reverse=True)
        bottom_returns = sorted(normalized_results, key=lambda x: x['return_pct'])
        
        # 创建 stock_code 到 result 的映射，方便模板访问
        results_dict = {r['stock_code']: r for r in normalized_results}
        
        # 将 stock_details 与 results 合并，方便模板访问
        merged_stock_details = []
        for detail in stock_details:
            stock_code = detail['stock_code']
            if stock_code in results_dict:
                merged_detail = detail.copy()
                merged_detail['result'] = results_dict[stock_code]
                merged_stock_details.append(merged_detail)
        
        # 准备模板数据
        template_data = {
            'title': f"{strategy_name} - 多股票回测综合报告",
            'strategy_name': strategy_name,
            'start_date': start_date,
            'end_date': end_date,
            'stock_count': total_count,
            'initial_cash_per_stock': initial_cash_per_stock,
            'successful_count': successful_count,
            'total_count': total_count,
            'total_initial_cash': total_initial_cash,
            'total_final_equity': total_final_equity,
            'total_profit': total_profit,
            'avg_return': avg_return,
            'total_trades': total_trades,
            'results': normalized_results,
            'stock_details': merged_stock_details,  # 每个股票的详细信息（已合并 result）
            'top_returns': top_returns,
            'bottom_returns': bottom_returns,
            'failed_results': failed_results,
            'strategy_params': strategy_params or {},
            'report_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        # 渲染模板
        template = self.jinja_env.get_template('multi_stock_report_template.md.j2')
        markdown_report = template.render(**template_data)
        
        # 保存报告（使用简单的文件名 report.md）
        report_file = self.output_dir / "report.md"
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(markdown_report)
        
        logger.info(f"多股票综合报告已保存: {report_file}")
        return report_file
    
    def _generate_stock_detail(
        self,
        result: Dict,
        assets_dir: Path,
        start_date: str,
        end_date: str
    ) -> Optional[Dict]:
        """
        为单个股票生成图表和详细指标
        
        Args:
            result: 股票回测结果字典（包含 account, engine, market 等信息）
            assets_dir: assets 文件夹路径
            start_date: 开始日期
            end_date: 结束日期
        
        Returns:
            包含图表路径和详细指标的字典
        """
        try:
            account = result.get('account')
            engine = result.get('engine')
            market = result.get('market')
            stock_code = result['stock_code']
            actual_start_date = result.get('start_date', start_date)
            actual_end_date = result.get('end_date', end_date)
            
            if not account or not engine:
                logger.warning(f"股票 {stock_code} 缺少账户或引擎信息，跳过图表生成")
                return None
            
            # 创建临时的 BacktestReport 实例来生成图表
            # 注意：不传入 title，避免创建额外的子目录，图表直接保存到 assets_dir
            temp_report = BacktestReport(output_dir=assets_dir, title=None)
            temp_report.daily_records = engine.report.daily_records if engine.report else []
            temp_report.train_test_split_date = engine.train_test_split_date if engine else None
            
            if not temp_report.daily_records:
                logger.warning(f"股票 {stock_code} 没有每日记录，跳过图表生成")
                return None
            
            # 生成图表（会保存到 assets_dir）
            chart_file = temp_report._generate_charts(
                stock_code, actual_start_date, actual_end_date, account=account
            )
            
            # 如果生成了图表，重命名为标准格式
            chart_path = None
            if chart_file and chart_file.exists():
                # 重命名图表为标准格式
                new_chart_name = f"{stock_code.replace('.', '_')}_chart.png"
                new_chart_path = assets_dir / new_chart_name
                if chart_file != new_chart_path:
                    # 如果文件已存在，先删除
                    if new_chart_path.exists():
                        new_chart_path.unlink()
                    chart_file.rename(new_chart_path)
                chart_path = f"assets/{new_chart_name}"
            elif chart_file:
                # 如果图表文件路径存在但文件不存在，尝试使用文件名
                chart_path = f"assets/{chart_file.name}"
            
            # 计算详细指标
            df = pd.DataFrame(temp_report.daily_records)
            if df.empty:
                return {
                    'chart_path': chart_path,
                    'statistics': {}
                }
            
            df['date'] = pd.to_datetime(df['date'])
            df = df.sort_values('date')
            
            # 分离训练期和测试期
            if 'is_test_period' in df.columns:
                test_df = df[df['is_test_period']].copy()
            else:
                test_df = df.copy()
            
            if test_df.empty:
                test_df = df.copy()
            
            # 计算指标
            equities = test_df['equity'].tolist()
            returns = test_df['return_pct'].tolist()
            
            if len(equities) < 2:
                return {
                    'chart_path': chart_path,
                    'statistics': {}
                }
            
            # 计算最大回撤
            max_drawdown = temp_report._calculate_max_drawdown(equities)
            
            # 计算日收益率
            daily_returns = []
            for i in range(1, len(equities)):
                if equities[i-1] > 0:
                    ret = (equities[i] / equities[i-1]) - 1.0
                    daily_returns.append(ret)
            
            # 计算夏普比率
            sharpe_info = None
            if daily_returns:
                mean_return = sum(daily_returns) / len(daily_returns)
                variance = sum((r - mean_return) ** 2 for r in daily_returns) / (len(daily_returns) - 1) if len(daily_returns) > 1 else 0.0
                std_return = math.sqrt(variance) if variance > 0 else 0.0
                sharpe_daily = (mean_return / std_return) if std_return > 0 else 0.0
                sharpe_annual = sharpe_daily * math.sqrt(252)
                sharpe_info = {
                    'sharpe_daily': sharpe_daily,
                    'sharpe_annual': sharpe_annual
                }
            
            # 计算 CAGR
            start_equity = equities[0]
            end_equity = equities[-1]
            trading_days = len(equities)
            if start_equity > 0 and trading_days > 0:
                total_return = (end_equity / start_equity) - 1.0
                years = trading_days / 252.0
                cagr = ((1 + total_return) ** (1 / years) - 1) * 100 if years > 0 else 0.0
            else:
                cagr = 0.0
            
            # 计算 Calmar 比率
            calmar = (cagr / abs(max_drawdown)) if max_drawdown != 0 else None
            
            # 计算波动率
            volatility_annual = None
            if daily_returns and len(daily_returns) > 1:
                std_return = math.sqrt(sum((r - sum(daily_returns)/len(daily_returns))**2 for r in daily_returns) / (len(daily_returns) - 1))
                volatility_annual = std_return * math.sqrt(252) * 100
            
            # 计算 Sortino 比率
            sortino_info = None
            if daily_returns:
                sortino_info = temp_report._calculate_sortino(daily_returns, risk_free_rate_annual=0.0)
            
            # 计算尾部风险
            var_cvar = None
            if daily_returns:
                var_cvar = temp_report._calculate_tail_risk(daily_returns)
            
            # 计算最终持仓信息
            final_record = test_df.iloc[-1] if not test_df.empty else None
            final_positions = {}
            if final_record is not None and 'positions' in final_record and final_record['positions']:
                final_positions = final_record['positions']
            
            # 计算资金比（现金/权益）
            cash_ratio = None
            if final_record is not None:
                final_equity = final_record.get('equity', 0)
                final_cash = final_record.get('cash', 0)
                cash_ratio = (final_cash / final_equity * 100) if final_equity > 0 else 0.0
            
            statistics = {
                'max_drawdown_pct': max_drawdown,
                'cagr': cagr,
                'calmar': calmar,
                'volatility_annual': volatility_annual,
                'cash_ratio': cash_ratio,
                'trading_days': trading_days,
                'max_equity': max(equities),
                'min_equity': min(equities),
                'max_return_pct': max(returns),
                'min_return_pct': min(returns)
            }
            
            if sharpe_info:
                statistics.update(sharpe_info)
            
            if sortino_info:
                statistics['sortino_annual'] = sortino_info.get('sortino_annual', 0.0)
                statistics['sortino_daily'] = sortino_info.get('sortino_daily', 0.0)
            
            if var_cvar:
                statistics.update(var_cvar)
            
            # 只生成图表，不生成详细报告
            return {
                'chart_path': chart_path,
                'statistics': statistics,
                'final_positions': final_positions,
                'final_cash': final_record.get('cash', 0) if final_record is not None else 0,
                'final_equity': final_record.get('equity', 0) if final_record is not None else 0
            }
            
        except Exception as e:
            logger.error(f"生成股票 {result.get('stock_code', 'unknown')} 的详细指标时出错: {e}", exc_info=True)
            return None

