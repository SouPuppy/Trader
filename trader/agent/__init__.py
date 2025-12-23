"""
TradingAgent 模块
提供交易代理的接口和实现
"""
from trader.agent.TradingAgent import TradingAgent
from trader.agent.agent_dca import DCAAgent
from trader.agent.agent_turtle import TurtleAgent
from trader.agent.agent_dummy import DummyAgent
from trader.agent.agent_dummy_with_simple_risk_control import DummyAgentWithSimpleRiskControl
from trader.agent.agent_logistic import LogisticAgent
from trader.agent.MultiAssetTradingAgent import MultiAssetTradingAgent
from trader.agent.multiagent_weight_normalized import normalize_weights, combine_agent_weights

# 为了向后兼容，保留 AbstractAgent 作为别名
AbstractAgent = TradingAgent

__all__ = [
    'TradingAgent', 
    'AbstractAgent', 
    'DCAAgent', 
    'TurtleAgent',
    'DummyAgent',
    'DummyAgentWithSimpleRiskControl',
    'LogisticAgent',
    'MultiAssetTradingAgent',
    'normalize_weights',
    'combine_agent_weights',
]

