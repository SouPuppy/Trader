"""
TradingAgent 模块
提供交易代理的接口和实现
"""
from trader.agent.TradingAgent import TradingAgent
from trader.agent.abstract_agent import AbstractAgent
from trader.agent.dummy_agent import DummyAgent

__all__ = ['TradingAgent', 'AbstractAgent', 'DummyAgent']

