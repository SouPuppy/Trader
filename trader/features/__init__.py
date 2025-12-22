"""
特征模块
提供特征注册和管理功能
"""
from trader.features.registry import FeatureSpec, FEATURES, register, get_all_features, get_feature_names, get_feature, has_feature

# 导入特征定义以触发注册
from trader.features import features  # noqa: F401

__all__ = [
    'FeatureSpec', 
    'FEATURES', 
    'register', 
    'get_all_features',
    'get_feature_names',
    'get_feature',
    'has_feature',
]

