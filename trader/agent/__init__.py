"""
TradingAgent 模块
提供交易代理的接口和实现
"""
from trader.agent.TradingAgent import TradingAgent
from trader.agent.agent_dca import DCAAgent
from trader.agent.agent_turtle import TurtleAgent
from trader.agent.agent_chasing_extremes import ChasingExtremesAgent
from trader.agent.MultiAssetTurtleAgent import MultiAssetTurtleAgent
from trader.agent.multiagent_weight_normalized import normalize_weights, combine_agent_weights

# 为了向后兼容，保留 AbstractAgent 作为别名
AbstractAgent = TradingAgent

# 注意：
# - LogisticAgent 依赖 scikit-learn；为了避免“只用 Predictor/Turtle 也必须装 sklearn”，
#   这里改为惰性导入（lazy import）。
# - MultiAssetLogisticAgent / MultiAssetTradingAgent 同样依赖 LogisticAgent，也做惰性导入。
def __getattr__(name: str):
    if name == "LogisticAgent":
        from trader.agent.agent_logistic import LogisticAgent as _LogisticAgent
        return _LogisticAgent
    if name in ("MultiAssetLogisticAgent", "MultiAssetTradingAgent"):
        from trader.agent.MultiAssetTradingAgent import MultiAssetLogisticAgent as _MultiAssetLogisticAgent
        return _MultiAssetLogisticAgent
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

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

