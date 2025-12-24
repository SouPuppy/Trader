"""
RAG 系统测试示例
演示如何使用 RAG 系统回答问题
"""
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# 注意：由于 Python 导入机制的限制，需要直接从 answer 模块导入
# 如果从 trader.rag 导入，可能会导入模块而不是函数
from trader.rag.answer import rag_answer
from trader.agent.agent_turtle import TurtleAgent
from trader.backtest.account import Account
from trader.backtest.market import Market
from trader.backtest.engine import BacktestEngine
from trader.logger import get_logger

logger = get_logger(__name__)


def run_turtle_backtest_for_tsla():
    """
    运行 Turtle 策略回测来生成 trade_history 记录
    
    这个函数会在 RAG 测试之前运行，为 TSLA.O 创建交易记录
    """
    logger.info("=" * 80)
    logger.info("Running Turtle Strategy Backtest for TSLA.O to generate trade_history")
    logger.info("=" * 80)
    
    # 先清空 trade_history 表
    import sqlite3
    from trader.config import DB_PATH
    
    logger.info("Clearing trade_history table...")
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    try:
        cursor.execute("DELETE FROM trade_history")
        conn.commit()
        logger.info("trade_history table cleared successfully")
    except Exception as e:
        logger.warning(f"Failed to clear trade_history table: {e}")
    finally:
        conn.close()
    
    stock_code = "TSLA.O"
    initial_cash = 1000000.0
    
    # 初始化市场、账户和回测引擎
    market = Market(price_adjustment=0.01)
    account = Account(initial_cash=initial_cash)
    
    # 启用 trade_history 记录，不清空现有记录（因为已经手动清空了）
    # 使用 2023 年的数据，设置时间范围确保有足够的数据
    engine = BacktestEngine(
        account, 
        market, 
        enable_report=False,  # 不需要生成报告
        record_trade_history=True,  # 启用交易历史记录
        clear_trade_history=False,  # 不清空现有记录（已经手动清空了）
        train_test_split_ratio=0.0  # 不使用训练/测试分割，使用所有数据
    )
    
    # 获取可用日期
    available_dates = market.get_available_dates(stock_code)
    if not available_dates:
        logger.error(f"未找到股票 {stock_code} 的数据")
        return False
    
    logger.info(f"Available dates for {stock_code}: {available_dates[0]} to {available_dates[-1]}")
    
    # 创建海龟策略 Agent
    agent = TurtleAgent(
        name="Turtle_Strategy_TSLA",
        entry_period=20,
        exit_period=10,
        atr_period=20,
        risk_per_trade=0.02,
        stop_loss_atr=2.0,
        max_positions=4,
        add_position_atr=0.5
    )
    
    # 设置要交易的股票代码
    agent.set_trading_stocks([stock_code])
    
    # 注册交易日回调
    def on_trading_day(eng: BacktestEngine, date: str):
        """每个交易日的回调"""
        agent.on_date(eng, date)
    
    engine.on_date(on_trading_day)
    
    # 运行回测（使用 2023 年的数据）
    start_date = "2023-01-01"
    end_date = "2023-12-31"
    
    logger.info(f"Running backtest from {start_date} to {end_date}...")
    engine.run(stock_code, start_date=start_date, end_date=end_date)
    
    # 检查生成了多少交易记录
    import sqlite3
    from trader.config import DB_PATH
    
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM trade_history WHERE stock_code = ?", (stock_code,))
    trade_count = cursor.fetchone()[0]
    conn.close()
    
    logger.info(f"Generated {trade_count} trade records for {stock_code}")
    logger.info(f"Total trades executed: {len(account.trades)}")
    logger.info("=" * 80)
    logger.info("")
    
    return trade_count > 0


def main():
    """Main function"""
    print("=" * 80)
    print("RAG System Test Examples")
    print("=" * 80)
    print()
    
    # Step 1: Run Turtle backtest to generate trade_history for TSLA.O
    logger.info("Step 1: Generating trade_history records...")
    success = run_turtle_backtest_for_tsla()
    if not success:
        logger.warning("Failed to generate trade_history records. Some tests may fail.")
    print()
    
    # Store results for statistics
    results = []
    
    # Example 1: Market state query
    print("Example 1: Market State Query")
    print("-" * 80)
    question1 = "What is the market trend of AAPL.O in the last 30 days?"
    result1 = rag_answer(
        question=question1,
        stock_code="AAPL.O",
        decision_time="2023-12-15T00:00:00",
        frequency="1d"
    )
    print(f"\nQuestion: {question1}")
    print(f"\nAnswer:\n{result1.answer}")
    print(f"\nVerification Status: {'Passed' if result1.passed else 'Failed'}")
    print(f"Mode: {result1.mode}")
    if result1.violations:
        print(f"Violations: {result1.violations}")
    results.append(("Example 1: Market State Query", result1.passed, result1.mode, len(result1.violations)))
    print()
    
    # Example 2: News impact query
    print("Example 2: News Impact Query")
    print("-" * 80)
    question2 = "What are the important news about TSLA.O recently? What impact do they have on the stock price?"
    result2 = rag_answer(
        question=question2,
        stock_code="TSLA.O",
        decision_time="2023-12-15T00:00:00",
        frequency="1d"
    )
    print(f"\nQuestion: {question2}")
    print(f"\nAnswer:\n{result2.answer}")
    print(f"\nVerification Status: {'Passed' if result2.passed else 'Failed'}")
    print(f"Mode: {result2.mode}")
    results.append(("Example 2: News Impact Query", result2.passed, result2.mode, len(result2.violations)))
    print()
    
    # Example 3: Trading history query
    print("Example 3: Trading History Query")
    print("-" * 80)
    question3 = "What is the recent trading history of TSLA.O?"
    result3 = rag_answer(
        question=question3,
        stock_code="TSLA.O",
        decision_time="2023-12-15T00:00:00",
        frequency="1d"
    )
    print(f"\nQuestion: {question3}")
    print(f"\nAnswer:\n{result3.answer}")
    print(f"\nVerification Status: {'Passed' if result3.passed else 'Failed'}")
    print(f"Mode: {result3.mode}")
    results.append(("Example 3: Trading History Query", result3.passed, result3.mode, len(result3.violations)))
    print()
    
    # Example 4: Risk check
    print("Example 4: Risk Check")
    print("-" * 80)
    question4 = "What are the current risks for NVDA.O?"
    result4 = rag_answer(
        question=question4,
        stock_code="NVDA.O",
        decision_time="2023-12-15T00:00:00",
        frequency="1d"
    )
    print(f"\nQuestion: {question4}")
    print(f"\nAnswer:\n{result4.answer}")
    print(f"\nVerification Status: {'Passed' if result4.passed else 'Failed'}")
    print(f"Mode: {result4.mode}")
    results.append(("Example 4: Risk Check", result4.passed, result4.mode, len(result4.violations)))
    print()
    
    # Example 5: Trend analysis query
    print("Example 5: Trend Analysis Query")
    print("-" * 80)
    question5 = "Has AAPL.O been rising recently?"
    result5 = rag_answer(
        question=question5,
        stock_code="AAPL.O",
        decision_time="2023-12-15T00:00:00",
        frequency="1d"
    )
    print(f"\nQuestion: {question5}")
    print(f"\nAnswer:\n{result5.answer}")
    print(f"\nVerification Status: {'Passed' if result5.passed else 'Failed'}")
    print(f"Mode: {result5.mode}")
    if result5.violations:
        print(f"Violations: {result5.violations}")
    results.append(("Example 5: Trend Analysis Query", result5.passed, result5.mode, len(result5.violations)))
    print()
    
    # Example 6: TSLA.O Trading History Query
    print("Example 6: TSLA.O Trading History Query")
    print("-" * 80)
    question6 = "What is the recent trading history of TSLA.O? Show me the buy and sell actions."
    result6 = rag_answer(
        question=question6,
        stock_code="TSLA.O",
        decision_time="2023-12-15T00:00:00",
        frequency="1d"
    )
    print(f"\nQuestion: {question6}")
    print(f"\nAnswer:\n{result6.answer}")
    print(f"\nVerification Status: {'Passed' if result6.passed else 'Failed'}")
    print(f"Mode: {result6.mode}")
    results.append(("Example 6: TSLA.O Trading History Query", result6.passed, result6.mode, len(result6.violations)))
    print()
    
    # Example 7: NVDA.O Feature Analysis
    print("Example 7: NVDA.O Feature Analysis")
    print("-" * 80)
    question7 = "What are the key technical features of NVDA.O? Analyze the return rates, volatility, and financial ratios."
    result7 = rag_answer(
        question=question7,
        stock_code="NVDA.O",
        decision_time="2023-12-15T00:00:00",
        frequency="1d"
    )
    print(f"\nQuestion: {question7}")
    print(f"\nAnswer:\n{result7.answer}")
    print(f"\nVerification Status: {'Passed' if result7.passed else 'Failed'}")
    print(f"Mode: {result7.mode}")
    results.append(("Example 7: NVDA.O Feature Analysis", result7.passed, result7.mode, len(result7.violations)))
    print()
    
    # Example 8: NVDA.O News and Trading Correlation
    print("Example 8: NVDA.O News and Trading Correlation")
    print("-" * 80)
    question8 = "What news events happened around NVDA.O trading actions? Are there correlations between news sentiment and trading decisions?"
    result8 = rag_answer(
        question=question8,
        stock_code="NVDA.O",
        decision_time="2023-12-15T00:00:00",
        frequency="1d"
    )
    print(f"\nQuestion: {question8}")
    print(f"\nAnswer:\n{result8.answer}")
    print(f"\nVerification Status: {'Passed' if result8.passed else 'Failed'}")
    print(f"Mode: {result8.mode}")
    results.append(("Example 8: NVDA.O News and Trading Correlation", result8.passed, result8.mode, len(result8.violations)))
    print()
    
    # Example 9: NVDA.O Market Trend and Volatility
    print("Example 9: NVDA.O Market Trend and Volatility")
    print("-" * 80)
    question9 = "What is the market trend and volatility pattern of NVDA.O in the past month? How does it compare to historical patterns?"
    result9 = rag_answer(
        question=question9,
        stock_code="NVDA.O",
        decision_time="2023-12-15T00:00:00",
        frequency="1d"
    )
    print(f"\nQuestion: {question9}")
    print(f"\nAnswer:\n{result9.answer}")
    print(f"\nVerification Status: {'Passed' if result9.passed else 'Failed'}")
    print(f"Mode: {result9.mode}")
    results.append(("Example 9: NVDA.O Market Trend and Volatility", result9.passed, result9.mode, len(result9.violations)))
    print()
    
    # Example 10: NVDA.O Comprehensive Decision Support
    print("Example 10: NVDA.O Comprehensive Decision Support")
    print("-" * 80)
    question10 = "Based on recent trading history, news events, and technical features, what is the current market situation for NVDA.O? Should I consider buying or selling?"
    result10 = rag_answer(
        question=question10,
        stock_code="NVDA.O",
        decision_time="2023-12-15T00:00:00",
        frequency="1d"
    )
    print(f"\nQuestion: {question10}")
    print(f"\nAnswer:\n{result10.answer}")
    print(f"\nVerification Status: {'Passed' if result10.passed else 'Failed'}")
    print(f"Mode: {result10.mode}")
    if result10.violations:
        print(f"Violations: {result10.violations}")
    results.append(("Example 10: NVDA.O Comprehensive Decision Support", result10.passed, result10.mode, len(result10.violations)))
    print()
    
    # Calculate statistics
    print("=" * 80)
    print("Test Completed - All 10 Examples Executed")
    print("=" * 80)
    print()
    
    # Calculate pass/fail statistics
    total_tests = len(results)
    passed_tests = sum(1 for _, passed, _, _ in results if passed)
    failed_tests = total_tests - passed_tests
    pass_rate = (passed_tests / total_tests * 100) if total_tests > 0 else 0.0
    fail_rate = (failed_tests / total_tests * 100) if total_tests > 0 else 0.0
    
    # Count by mode
    normal_mode = sum(1 for _, _, mode, _ in results if mode == "normal")
    degraded_mode = sum(1 for _, _, mode, _ in results if mode == "degraded")
    
    # Count violations
    total_violations = sum(violations for _, _, _, violations in results)
    tests_with_violations = sum(1 for _, _, _, violations in results if violations > 0)
    
    print("=" * 80)
    print("Test Statistics Summary")
    print("=" * 80)
    print(f"Total Tests: {total_tests}")
    print(f"Passed: {passed_tests} ({pass_rate:.1f}%)")
    print(f"Failed: {failed_tests} ({fail_rate:.1f}%)")
    print()
    print(f"Mode Distribution:")
    print(f"  Normal: {normal_mode} ({normal_mode/total_tests*100:.1f}%)")
    print(f"  Degraded: {degraded_mode} ({degraded_mode/total_tests*100:.1f}%)")
    print()
    print(f"Violations:")
    print(f"  Total Violations: {total_violations}")
    print(f"  Tests with Violations: {tests_with_violations} ({tests_with_violations/total_tests*100:.1f}%)")
    print()
    print("Detailed Results:")
    print("-" * 80)
    for test_name, passed, mode, violations in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        violation_str = f" ({violations} violations)" if violations > 0 else ""
        print(f"  {status} | {mode:8s} | {test_name}{violation_str}")
    print("=" * 80)


if __name__ == "__main__":
    main()

