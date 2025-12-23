"""
Dataloader 模块
提供从开始日期到结束日期的所有 features 加载功能
支持原始数据、null 补全、自动插值补全、前向填充和线性插值五种模式
"""
from trader.dataloader.dataloader import Dataloader
from trader.dataloader.dataloader_raw import dataloader_raw
from trader.dataloader.dataloader_nullcomplete import dataloader_nullcomplete
from trader.dataloader.dataloader_autocomplete import dataloader_autocomplete
from trader.dataloader.dataloader_ffill import dataloader_ffill
from trader.dataloader.dataloader_linear import dataloader_linear

__all__ = [
    'Dataloader',
    'dataloader_raw',
    'dataloader_nullcomplete',
    'dataloader_autocomplete',
    'dataloader_ffill',
    'dataloader_linear',
]
