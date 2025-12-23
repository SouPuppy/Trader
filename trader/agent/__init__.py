"""
TradingAgent 模块
提供交易代理的接口和实现
"""
from trader.agent.TradingAgent import TradingAgent
from trader.agent.dummy_agent import DummyAgent

# 为了向后兼容，保留 AbstractAgent 作为别名
AbstractAgent = TradingAgent

__all__ = ['TradingAgent', 'AbstractAgent', 'DummyAgent']

