"""
特征注册表
提供特征注册和管理功能
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional
import pandas as pd

logger = None


def _get_logger():
    """延迟导入 logger，避免循环依赖"""
    global logger
    if logger is None:
        from trader.logger import get_logger
        logger = get_logger(__name__)
    return logger


@dataclass(frozen=True)
class FeatureSpec:
    """
    特征规范定义
    
    Attributes:
        name: 特征名称
        dtype: 数据类型（如 'float64', 'int64'）
        lookback: 回看窗口大小（用于时间序列特征，0 表示只使用当前值）
        compute: 计算函数，接受 pd.DataFrame 返回 pd.Series
        deps: 依赖的其他特征名称列表（可选）
        desc: 特征描述（可选）
    """
    name: str
    dtype: str
    lookback: int
    compute: Callable[[pd.DataFrame], pd.Series]
    deps: Optional[List[str]] = None
    desc: str = ""

    def __post_init__(self):
        """验证特征规范"""
        if self.lookback < 0:
            raise ValueError(f"lookback must be >= 0, got {self.lookback}")
        if self.deps is not None:
            for dep in self.deps:
                if not isinstance(dep, str):
                    raise ValueError(f"deps must be a list of strings, got {dep}")


# 全局特征注册表
FEATURES: Dict[str, FeatureSpec] = {}


def register(spec: FeatureSpec) -> FeatureSpec:
    """
    注册一个特征
    
    Args:
        spec: 特征规范
        
    Returns:
        注册的特征规范
        
    Raises:
        ValueError: 如果特征名称已存在
    """
    if spec.name in FEATURES:
        raise ValueError(f"Feature already registered: {spec.name}")
    
    # 验证依赖关系
    if spec.deps:
        missing_deps = [dep for dep in spec.deps if dep not in FEATURES]
        if missing_deps:
            _get_logger().warning(
                f"Feature {spec.name} has unresolved dependencies: {missing_deps}. "
                "Make sure to register dependencies first."
            )
    
    FEATURES[spec.name] = spec
    _get_logger().debug(f"Registered feature: {spec.name}")
    return spec


def get_feature(name: str) -> Optional[FeatureSpec]:
    """
    根据名称获取特征规范
    
    Args:
        name: 特征名称
        
    Returns:
        特征规范，如果不存在则返回 None
    """
    return FEATURES.get(name)


def get_all_features() -> List[FeatureSpec]:
    """
    获取所有已注册的特征
    
    Returns:
        特征规范列表
    """
    return list(FEATURES.values())


def get_feature_names() -> List[str]:
    """
    获取所有已注册的特征名称
    
    Returns:
        特征名称列表
    """
    return list(FEATURES.keys())


def has_feature(name: str) -> bool:
    """
    检查特征是否已注册
    
    Args:
        name: 特征名称
        
    Returns:
        如果特征已注册返回 True，否则返回 False
    """
    return name in FEATURES

