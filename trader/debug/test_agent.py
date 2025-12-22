"""
测试 agent 接口的使用示例
演示 score 和 weight 的区别和使用
"""
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from trader.agent import DummyAgent
from trader.backtest.account import Account
from trader.backtest.market import Market
from trader.backtest.engine import BacktestEngine
from trader.logger import get_logger, log_separator, log_section

logger = get_logger(__name__)


def test_agent():
    """测试 agent 的使用"""
    for line in log_section("测试 agent 接口"):
        logger.info(line)
    
    # 初始化
    market = Market(price_adjustment=0.01)
    account = Account(initial_cash=100000.0)
    engine = BacktestEngine(account, market, enable_report=False)
    
    # 创建 Dummy agent
    agent = DummyAgent(
        name="TestAgent",
        max_position_weight=0.1,  # 单个股票最多配置10%
        min_score_threshold=0.0,   # score >= 0 才考虑配置
        max_total_weight=0.8       # 总配置不超过80%
    )
    
    stock_code = "AAPL.O"
    stock_codes = [stock_code]
    
    def on_trading_day(eng: BacktestEngine, date: str):
        """每个交易日的回调"""
        if eng.date_index == 0:  # 只在第一天演示
            logger.info(f"\n[{date}] agent 使用示例:")
            
            # 1. 计算 score（研究问题：看好程度）
            score = agent.score(stock_code, eng)
            logger.info(f"  Score ({stock_code}): {score:.4f}")
            logger.info(f"    - Score 表示看好程度/预期收益")
            logger.info(f"    - 范围: [-1, 1]，正数表示看好，负数表示看空")
            
            # 2. 计算 weight（工程+风控问题：实际配置比例）
            weight = agent.weight(stock_code, score, eng)
            logger.info(f"  Weight ({stock_code}): {weight:.4f} ({weight*100:.2f}%)")
            logger.info(f"    - Weight 表示实际资金配置比例")
            logger.info(f"    - 范围: [0, max_position_weight]")
            logger.info(f"    - 考虑了风险控制和仓位限制")
            
            # 3. 批量计算
            scores = agent.get_scores(stock_codes, eng)
            weights = agent.get_weights(scores, eng)
            logger.info(f"\n  批量计算:")
            logger.info(f"    Scores: {scores}")
            logger.info(f"    Weights: {weights}")
            
            # 4. 演示 score 和 weight 的区别
            logger.info(f"\n  Score vs Weight 的区别:")
            logger.info(f"    - Score: 研究问题，表达'看好程度'")
            logger.info(f"    - Weight: 工程+风控问题，表达'实际配置多少资金'")
            logger.info(f"    - 即使 score 很高，weight 也可能因为风控而较小")
    
    engine.on_date(on_trading_day)
    
    # 运行回测（只运行1天）
    available_dates = market.get_available_dates(stock_code)
    if available_dates:
        start_date = available_dates[0]
        engine.run(stock_code, start_date=start_date, end_date=start_date)
    
    logger.info("")
    for line in log_section("测试完成"):
        logger.info(line)


if __name__ == "__main__":
    test_agent()

