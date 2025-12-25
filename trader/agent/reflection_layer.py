"""
Layer C: Reflection Layer（反思层）
低频读取近期交易表现与证据，更新执行层的风险与仓位参数θ
"""
from typing import Dict, Optional, List
from datetime import datetime, timedelta
from trader.agent.theta import Theta
from trader.backtest.report import BacktestReport
from trader.logger import get_logger
import json

logger = get_logger(__name__)


class ReflectionPacket:
    """
    反思包：包含交易表现摘要和证据
    """
    
    def __init__(
        self,
        performance_summary: Dict,
        portfolio_state: Dict,
        evidence_docs: List[Dict],
        reflection_date: str
    ):
        """
        初始化反思包
        
        Args:
            performance_summary: 交易表现摘要
            portfolio_state: 组合状态摘要
            evidence_docs: 证据文档列表
            reflection_date: 反思日期
        """
        self.performance_summary = performance_summary
        self.portfolio_state = portfolio_state
        self.evidence_docs = evidence_docs
        self.reflection_date = reflection_date
    
    def to_dict(self) -> Dict:
        """转换为字典"""
        return {
            "performance_summary": self.performance_summary,
            "portfolio_state": self.portfolio_state,
            "evidence_docs": self.evidence_docs,
            "reflection_date": self.reflection_date
        }


class ReflectionLayer:
    """
    反思层：定期反思交易表现，调整参数θ
    """
    
    def __init__(
        self,
        reflection_frequency_days: int = 5,  # 每5个交易日反思一次
        lookback_days: int = 20,  # 回顾最近20天的表现
        llm_model: str = "deepseek-chat",
        llm_temperature: float = 0.3
    ):
        """
        初始化反思层
        
        Args:
            reflection_frequency_days: 反思频率（交易日数）
            lookback_days: 回顾天数
            llm_model: LLM模型名称
            llm_temperature: LLM温度参数
        """
        self.reflection_frequency_days = reflection_frequency_days
        self.lookback_days = lookback_days
        self.llm_model = llm_model
        self.llm_temperature = llm_temperature
        
        self.last_reflection_date: Optional[str] = None
        self.reflection_count = 0
        self.theta_history: List[Theta] = []
    
    def should_reflect(self, current_date: str) -> bool:
        """
        判断是否应该进行反思
        
        Args:
            current_date: 当前日期
            
        Returns:
            bool: 是否应该反思
        """
        if self.last_reflection_date is None:
            return True
        
        # 计算距离上次反思的天数
        try:
            last_date = datetime.fromisoformat(self.last_reflection_date.replace("T", " "))
            current_date_dt = datetime.fromisoformat(current_date.replace("T", " "))
            days_diff = (current_date_dt - last_date).days
            
            return days_diff >= self.reflection_frequency_days
        except Exception as e:
            logger.warning(f"判断反思时机失败: {e}")
            return False
    
    def build_reflection_packet(
        self,
        report: BacktestReport,
        current_date: str,
        stock_codes: List[str]
    ) -> ReflectionPacket:
        """
        构建反思包
        
        Args:
            report: 回测报告
            current_date: 当前日期
            stock_codes: 股票代码列表
            
        Returns:
            ReflectionPacket: 反思包
        """
        # 1. 计算交易表现摘要（最近M天）
        performance_summary = self._calculate_performance_summary(
            report, current_date
        )
        
        # 2. 组合状态摘要
        portfolio_state = self._calculate_portfolio_state(
            report, current_date, stock_codes
        )
        
        # 3. 证据文档（使用RAG检索）
        evidence_docs = self._retrieve_evidence(
            current_date, stock_codes
        )
        
        return ReflectionPacket(
            performance_summary=performance_summary,
            portfolio_state=portfolio_state,
            evidence_docs=evidence_docs,
            reflection_date=current_date
        )
    
    def _calculate_performance_summary(
        self,
        report: BacktestReport,
        current_date: str
    ) -> Dict:
        """计算交易表现摘要"""
        if not report.daily_records:
            return {}
        
        # 获取最近M天的记录
        try:
            current_date_dt = datetime.fromisoformat(current_date.replace("T", " "))
            cutoff_date = current_date_dt - timedelta(days=self.lookback_days)
            
            recent_records = [
                r for r in report.daily_records
                if datetime.fromisoformat(r["date"].replace("T", " ")) >= cutoff_date
            ]
            
            if not recent_records:
                return {}
            
            # 计算指标
            equities = [r["equity"] for r in recent_records]
            returns = [r["return_pct"] for r in recent_records]
            
            # 总收益
            total_return = returns[-1] - returns[0] if len(returns) > 1 else 0.0
            
            # 最大回撤
            max_drawdown = 0.0
            peak = equities[0]
            for equity in equities:
                if equity > peak:
                    peak = equity
                drawdown = (peak - equity) / peak * 100 if peak > 0 else 0.0
                max_drawdown = max(max_drawdown, drawdown)
            
            # 波动率（简化）
            if len(returns) > 1:
                import statistics
                volatility = statistics.stdev(returns) * (252 ** 0.5)  # 年化
            else:
                volatility = 0.0
            
            # 换手率（简化，从交易记录估算）
            # 这里简化处理，实际应该从account.trades计算
            
            return {
                "total_return": total_return,
                "max_drawdown": max_drawdown,
                "volatility": volatility,
                "period_days": len(recent_records)
            }
        except Exception as e:
            logger.error(f"计算表现摘要失败: {e}", exc_info=True)
            return {}
    
    def _calculate_portfolio_state(
        self,
        report: BacktestReport,
        current_date: str,
        stock_codes: List[str]
    ) -> Dict:
        """计算组合状态摘要"""
        if not report.daily_records:
            return {}
        
        # 获取最新记录
        latest_record = report.daily_records[-1]
        
        # 计算总敞口
        positions_value = latest_record.get("positions_value", 0.0)
        equity = latest_record.get("equity", 0.0)
        gross_exposure = positions_value / equity if equity > 0 else 0.0
        
        # 计算集中度（top5权重之和）
        position_weights = latest_record.get("position_weights", {})
        sorted_weights = sorted(position_weights.values(), reverse=True)
        top5_concentration = sum(sorted_weights[:5])
        
        # 现金比例
        cash = latest_record.get("cash", 0.0)
        cash_ratio = cash / equity if equity > 0 else 1.0
        
        return {
            "gross_exposure": gross_exposure,
            "top5_concentration": top5_concentration,
            "cash_ratio": cash_ratio,
            "num_positions": len(position_weights)
        }
    
    def _retrieve_evidence(
        self,
        current_date: str,
        stock_codes: List[str]
    ) -> List[Dict]:
        """
        检索证据文档（使用RAG系统）
        
        Args:
            current_date: 当前日期
            stock_codes: 股票代码列表
            
        Returns:
            List[Dict]: 证据文档列表
        """
        # 这里简化处理，实际应该调用RAG系统
        # 返回空列表表示没有证据（非反思层版本）
        return []
    
    def reflect(
        self,
        packet: ReflectionPacket,
        current_theta: Theta
    ) -> Optional[Theta]:
        """
        执行反思，生成新的参数θ
        
        Args:
            packet: 反思包
            current_theta: 当前参数θ
            
        Returns:
            Optional[Theta]: 新的参数θ，如果反思失败返回None
        """
        # 这里简化处理，实际应该调用LLM生成参数调整
        # 对于非反思层版本，直接返回当前theta
        logger.info(f"[ReflectionLayer] 执行反思（第{self.reflection_count + 1}次）")
        logger.info(f"当前参数θ: {current_theta}")
        
        # 更新反思状态
        self.last_reflection_date = packet.reflection_date
        self.reflection_count += 1
        
        # 记录历史
        new_theta = current_theta.copy()
        new_theta.reflection_id = self.reflection_count
        new_theta.timestamp = datetime.now()
        self.theta_history.append(new_theta)
        
        # 返回当前theta（实际应该由LLM生成调整）
        return new_theta

