"""
TradingAgent 模块
提供交易代理的接口和实现
"""
from trader.agent.TradingAgent import TradingAgent
from trader.agent.agent_dca import DCAAgent
from trader.agent.agent_turtle import TurtleAgent
from trader.agent.agent_logistic import LogisticAgent
from trader.agent.agent_chasing_extremes import ChasingExtremesAgent
from trader.agent.MultiAssetTradingAgent import MultiAssetLogisticAgent
from trader.agent.MultiAssetTurtleAgent import MultiAssetTurtleAgent
from trader.agent.multiagent_weight_normalized import normalize_weights, combine_agent_weights

# 为了向后兼容，保留 AbstractAgent 作为别名
AbstractAgent = TradingAgent
# 向后兼容：MultiAssetTradingAgent 指向 MultiAssetLogisticAgent
MultiAssetTradingAgent = MultiAssetLogisticAgent

__all__ = [
    'TradingAgent', 
    'AbstractAgent', 
    'DCAAgent', 
    'TurtleAgent',
    'LogisticAgent',
    'ChasingExtremesAgent',
    'MultiAssetLogisticAgent',
    'MultiAssetTurtleAgent',
    'MultiAssetTradingAgent',  # 向后兼容别名
    'normalize_weights',
    'combine_agent_weights',
]

