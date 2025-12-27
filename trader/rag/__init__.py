"""
RAG 系统模块
所有 RAG 相关的日志输出到 log/rag.log
"""

# 延迟导入，避免循环导入问题
def get_rag_logger(name: str = None):
    """
    获取 RAG 系统的 logger 实例，所有输出到 log/rag.log
    
    这是一个包装函数，实际从 trader.logger 导入 get_rag_logger
    """
    # 使用延迟导入避免循环导入
    import trader.logger as logger_module
    return logger_module.get_rag_logger(name)

# 导出 RAG logger 函数，方便其他模块使用
__all__ = ['get_rag_logger']


