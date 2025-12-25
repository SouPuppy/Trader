"""
可训练的反思系统：带反馈循环和学习机制

核心设计：
1. 记录反思历史：每次参数调整、原因、调整后的表现
2. 评估调整效果：对比调整前后的表现指标
3. 学习机制：从历史经验中学习最优调整策略
4. 反馈循环：用学习到的知识改进未来的调整决策
"""
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime
from dataclasses import dataclass, asdict
import numpy as np
from trader.agent.theta import Theta
from trader.logger import get_logger

logger = get_logger(__name__)


@dataclass
class ReflectionRecord:
    """单次反思记录"""
    reflection_id: int
    date: str
    current_theta: Dict  # 调整前的参数
    new_theta: Dict  # 调整后的参数
    weekly_return_before: float  # 调整前一周的收益率
    weekly_return_after: float  # 调整后一周的收益率
    account_equity_before: float
    account_equity_after: float
    adjustment_reason: str  # 调整原因（RAG分析或naive规则）
    adjustment_source: str  # "rag", "naive", "learned"
    # 表现指标（调整后一周的表现）
    sharpe_ratio: Optional[float] = None
    max_drawdown: Optional[float] = None
    volatility: Optional[float] = None
    win_rate: Optional[float] = None
    
    def to_dict(self) -> Dict:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'ReflectionRecord':
        return cls(**data)


@dataclass
class AdjustmentEffectiveness:
    """调整效果评估"""
    reflection_id: int
    date: str
    # 调整效果指标
    return_improvement: float  # 收益率改善（调整后 - 调整前）
    sharpe_improvement: Optional[float] = None
    drawdown_change: Optional[float] = None
    # 调整质量评分（0-1，1表示完美调整）
    effectiveness_score: float = 0.0
    # 调整方向是否正确
    direction_correct: bool = False
    
    def to_dict(self) -> Dict:
        return asdict(self)


class LearnableReflectionSystem:
    """
    可训练的反思系统
    
    功能：
    1. 记录每次反思的历史
    2. 评估调整效果
    3. 从历史经验中学习
    4. 提供基于学习的参数调整建议
    """
    
    def __init__(self, history_file: Optional[Path] = None):
        """
        初始化可训练的反思系统
        
        Args:
            history_file: 历史记录文件路径（用于持久化）
        """
        self.history_file = history_file
        self.reflection_history: List[ReflectionRecord] = []
        self.effectiveness_history: List[AdjustmentEffectiveness] = []
        
        # 学习到的知识
        self.learned_patterns: Dict = {
            # 不同市场状态下的最优参数调整策略
            "market_states": {},
            # 不同收益率区间下的调整效果
            "return_ranges": {},
            # 参数调整方向的效果统计
            "adjustment_directions": {}
        }
        
        # 加载历史记录
        if history_file and history_file.exists():
            self.load_history()
    
    def record_reflection(
        self,
        reflection_id: int,
        date: str,
        current_theta: Theta,
        new_theta: Theta,
        weekly_return_before: float,
        account_equity_before: float,
        adjustment_reason: str,
        adjustment_source: str
    ):
        """
        记录一次反思
        
        Args:
            reflection_id: 反思ID
            date: 日期
            current_theta: 调整前的参数
            new_theta: 调整后的参数
            weekly_return_before: 调整前一周的收益率
            account_equity_before: 调整前的账户权益
            adjustment_reason: 调整原因
            adjustment_source: 调整来源（"rag", "naive", "learned"）
        """
        record = ReflectionRecord(
            reflection_id=reflection_id,
            date=date,
            current_theta=current_theta.to_dict(),
            new_theta=new_theta.to_dict(),
            weekly_return_before=weekly_return_before,
            weekly_return_after=0.0,  # 将在后续更新
            account_equity_before=account_equity_before,
            account_equity_after=account_equity_before,  # 将在后续更新
            adjustment_reason=adjustment_reason,
            adjustment_source=adjustment_source
        )
        self.reflection_history.append(record)
        
        # 保存到文件
        if self.history_file:
            self.save_history()
    
    def update_reflection_result(
        self,
        reflection_id: int,
        weekly_return_after: float,
        account_equity_after: float,
        sharpe_ratio: Optional[float] = None,
        max_drawdown: Optional[float] = None,
        volatility: Optional[float] = None,
        win_rate: Optional[float] = None
    ):
        """
        更新反思结果（在调整后一周结束时调用）
        
        Args:
            reflection_id: 反思ID
            weekly_return_after: 调整后一周的收益率
            account_equity_after: 调整后的账户权益
            sharpe_ratio: 夏普比率
            max_drawdown: 最大回撤
            volatility: 波动率
            win_rate: 胜率
        """
        # 找到对应的记录
        for record in self.reflection_history:
            if record.reflection_id == reflection_id:
                record.weekly_return_after = weekly_return_after
                record.account_equity_after = account_equity_after
                record.sharpe_ratio = sharpe_ratio
                record.max_drawdown = max_drawdown
                record.volatility = volatility
                record.win_rate = win_rate
                break
        
        # 评估调整效果
        self._evaluate_adjustment_effectiveness(reflection_id)
        
        # 更新学习到的知识
        self._update_learned_patterns(reflection_id)
        
        # 保存到文件
        if self.history_file:
            self.save_history()
    
    def _evaluate_adjustment_effectiveness(self, reflection_id: int):
        """评估调整效果"""
        record = None
        for r in self.reflection_history:
            if r.reflection_id == reflection_id:
                record = r
                break
        
        if not record or record.weekly_return_after == 0.0:
            return
        
        # 计算收益率改善
        return_improvement = record.weekly_return_after - record.weekly_return_before
        
        # 计算有效性评分（基于收益率改善）
        # 如果调整后收益率更好，评分更高
        if return_improvement > 0.01:  # 改善超过1%
            effectiveness_score = min(1.0, 0.5 + return_improvement * 10)
        elif return_improvement > 0:
            effectiveness_score = 0.5 + return_improvement * 5
        elif return_improvement > -0.01:
            effectiveness_score = 0.5 + return_improvement * 5
        else:  # 恶化超过1%
            effectiveness_score = max(0.0, 0.5 + return_improvement * 10)
        
        # 判断调整方向是否正确
        # 如果调整前收益率为负，调整后应该提高仓位（激进）
        # 如果调整前收益率为正，调整后可以保持或适度激进
        direction_correct = False
        if record.weekly_return_before < 0:
            # 负收益时，应该提高仓位（降低enter_th，提高gross_exposure）
            gross_change = record.new_theta["gross_exposure"] - record.current_theta["gross_exposure"]
            enter_change = record.new_theta["enter_th"] - record.current_theta["enter_th"]
            if gross_change > 0 or enter_change < 0:
                direction_correct = True
        else:
            # 正收益时，可以保持或适度激进
            direction_correct = True  # 暂时认为都是正确的
        
        effectiveness = AdjustmentEffectiveness(
            reflection_id=reflection_id,
            date=record.date,
            return_improvement=return_improvement,
            sharpe_improvement=None,  # 可以后续添加
            drawdown_change=None,  # 可以后续添加
            effectiveness_score=effectiveness_score,
            direction_correct=direction_correct
        )
        
        self.effectiveness_history.append(effectiveness)
    
    def _update_learned_patterns(self, reflection_id: int):
        """更新学习到的模式"""
        record = None
        effectiveness = None
        
        for r in self.reflection_history:
            if r.reflection_id == reflection_id:
                record = r
                break
        
        for e in self.effectiveness_history:
            if e.reflection_id == reflection_id:
                effectiveness = e
                break
        
        if not record or not effectiveness:
            return
        
        # 学习：不同收益率区间下的调整效果
        return_range = self._get_return_range(record.weekly_return_before)
        if return_range not in self.learned_patterns["return_ranges"]:
            self.learned_patterns["return_ranges"][return_range] = {
                "count": 0,
                "total_effectiveness": 0.0,
                "successful_adjustments": []
            }
        
        pattern = self.learned_patterns["return_ranges"][return_range]
        pattern["count"] += 1
        pattern["total_effectiveness"] += effectiveness.effectiveness_score
        
        # 记录成功的调整（降低阈值，让更多激进调整被学习）
        # 如果收益率改善 > 0 或者有效性评分 > 0.5，都认为是成功的
        if effectiveness.effectiveness_score > 0.5 or effectiveness.return_improvement > 0:
            successful_adjustment = {
                "theta_before": record.current_theta,
                "theta_after": record.new_theta,
                "effectiveness": effectiveness.effectiveness_score,
                "return_improvement": effectiveness.return_improvement
            }
            pattern["successful_adjustments"].append(successful_adjustment)
            # 按有效性评分排序，保留最好的15次成功调整（增加数量）
            pattern["successful_adjustments"].sort(key=lambda x: x["effectiveness"], reverse=True)
            if len(pattern["successful_adjustments"]) > 15:
                pattern["successful_adjustments"] = pattern["successful_adjustments"][:15]
    
    def _get_return_range(self, weekly_return: float) -> str:
        """获取收益率区间标签"""
        if weekly_return > 0.02:  # > 2%
            return "high_positive"
        elif weekly_return > 0.01:  # 1-2%
            return "positive"
        elif weekly_return > -0.01:  # -1% 到 1%
            return "neutral"
        elif weekly_return > -0.02:  # -2% 到 -1%
            return "negative"
        else:  # < -2%
            return "high_negative"
    
    def learn_adjustment(
        self,
        current_theta: Theta,
        weekly_return: float,
        account_equity: float
    ) -> Optional[Theta]:
        """
        基于学习到的知识生成参数调整建议
        
        Args:
            current_theta: 当前参数
            weekly_return: 周收益率
            account_equity: 账户权益
            
        Returns:
            调整后的参数，如果没有学习到有效模式则返回None
        """
        if len(self.effectiveness_history) < 3:
            # 历史数据不足，无法学习
            return None
        
        return_range = self._get_return_range(weekly_return)
        
        # 查找相似情况下的成功调整
        if return_range not in self.learned_patterns["return_ranges"]:
            return None
        
        pattern = self.learned_patterns["return_ranges"][return_range]
        
        if pattern["count"] < 2:  # 降低样本要求，从3降到2
            # 样本不足
            return None
        
        # 计算平均有效性
        avg_effectiveness = pattern["total_effectiveness"] / pattern["count"]
        
        if avg_effectiveness < 0.4:  # 降低阈值，从0.5降到0.4，让更多情况使用学习建议
            # 平均效果不佳，不推荐调整
            return None
        
        # 找到最相似的成功调整
        best_adjustment = None
        best_score = 0.0
        
        for adj in pattern["successful_adjustments"]:
            # 计算相似度（基于当前参数和调整前参数的差异）
            similarity = self._calculate_similarity(
                current_theta.to_dict(),
                adj["theta_before"]
            )
            score = similarity * adj["effectiveness"]
            
            if score > best_score:
                best_score = score
                best_adjustment = adj
        
        if not best_adjustment or best_score < 0.2:  # 降低相似度阈值，从0.3降到0.2
            return None
        
        # 基于最佳调整生成新参数
        learned_theta = Theta(
            gross_exposure=best_adjustment["theta_after"]["gross_exposure"],
            max_w=best_adjustment["theta_after"]["max_w"],
            turnover_cap=best_adjustment["theta_after"]["turnover_cap"],
            risk_mode=best_adjustment["theta_after"]["risk_mode"],
            enter_th=best_adjustment["theta_after"]["enter_th"],
            exit_th=best_adjustment["theta_after"]["exit_th"]
        )
        
        logger.info(f"[Learnable Reflection] 基于历史经验生成调整建议（相似度: {best_score:.2f}, 有效性: {best_adjustment['effectiveness']:.2f}）")
        
        return learned_theta
    
    def _calculate_similarity(self, theta1: Dict, theta2: Dict) -> float:
        """计算两个参数集的相似度"""
        # 简单的欧氏距离相似度
        keys = ["gross_exposure", "max_w", "turnover_cap", "enter_th", "exit_th"]
        distances = []
        
        for key in keys:
            if key in theta1 and key in theta2:
                diff = abs(theta1[key] - theta2[key])
                # 归一化到 [0, 1]
                if key == "gross_exposure":
                    normalized_diff = diff / 0.5  # 范围约 [0.4, 0.85]
                elif key == "max_w":
                    normalized_diff = diff / 0.1  # 范围约 [0.08, 0.18]
                elif key == "turnover_cap":
                    normalized_diff = diff / 0.25  # 范围约 [0.10, 0.35]
                elif key == "enter_th":
                    normalized_diff = diff / 0.04  # 范围约 [0.01, 0.05]
                elif key == "exit_th":
                    normalized_diff = diff / 0.07  # 范围约 [-0.15, -0.08]
                else:
                    normalized_diff = 0.0
                
                distances.append(normalized_diff)
        
        if not distances:
            return 0.0
        
        avg_distance = np.mean(distances)
        similarity = 1.0 - min(1.0, avg_distance)  # 转换为相似度
        
        return similarity
    
    def get_statistics(self) -> Dict:
        """获取学习统计信息"""
        stats = {
            "total_reflections": len(self.reflection_history),
            "total_effectiveness_evaluations": len(self.effectiveness_history),
            "avg_effectiveness": 0.0,
            "learned_patterns_count": len(self.learned_patterns["return_ranges"]),
            "return_ranges": {}
        }
        
        if self.effectiveness_history:
            stats["avg_effectiveness"] = np.mean([
                e.effectiveness_score for e in self.effectiveness_history
            ])
        
        for return_range, pattern in self.learned_patterns["return_ranges"].items():
            stats["return_ranges"][return_range] = {
                "count": pattern["count"],
                "avg_effectiveness": pattern["total_effectiveness"] / pattern["count"] if pattern["count"] > 0 else 0.0,
                "successful_adjustments_count": len(pattern["successful_adjustments"])
            }
        
        return stats
    
    def save_history(self):
        """保存历史记录到文件"""
        if not self.history_file:
            return
        
        data = {
            "reflection_history": [r.to_dict() for r in self.reflection_history],
            "effectiveness_history": [e.to_dict() for e in self.effectiveness_history],
            "learned_patterns": self.learned_patterns
        }
        
        with open(self.history_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"[Learnable Reflection] 保存历史记录到 {self.history_file}")
    
    def load_history(self):
        """从文件加载历史记录"""
        if not self.history_file or not self.history_file.exists():
            return
        
        try:
            with open(self.history_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            self.reflection_history = [
                ReflectionRecord.from_dict(r) for r in data.get("reflection_history", [])
            ]
            self.effectiveness_history = [
                AdjustmentEffectiveness(**e) for e in data.get("effectiveness_history", [])
            ]
            self.learned_patterns = data.get("learned_patterns", {
                "market_states": {},
                "return_ranges": {},
                "adjustment_directions": {}
            })
            
            logger.info(f"[Learnable Reflection] 从 {self.history_file} 加载了 {len(self.reflection_history)} 条历史记录")
        except Exception as e:
            logger.error(f"[Learnable Reflection] 加载历史记录失败: {e}", exc_info=True)

