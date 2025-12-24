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
from trader.logger import get_logger

logger = get_logger(__name__)


def main():
    """Main function"""
    print("=" * 80)
    print("RAG System Test Examples")
    print("=" * 80)
    print()
    
    # Example 1: Market state query
    print("Example 1: Market State Query")
    print("-" * 80)
    question1 = "What is the market trend of AAPL.O in the last 30 days?"
    result1 = rag_answer(
        question=question1,
        stock_code="AAPL.O",
        decision_time="2024-01-15T00:00:00",
        frequency="1d"
    )
    print(f"\nQuestion: {question1}")
    print(f"\nAnswer:\n{result1.answer}")
    print(f"\nVerification Status: {'Passed' if result1.passed else 'Failed'}")
    print(f"Mode: {result1.mode}")
    if result1.violations:
        print(f"Violations: {result1.violations}")
    print()
    
    # Example 2: News impact query
    print("Example 2: News Impact Query")
    print("-" * 80)
    question2 = "What are the important news about TSLA.O recently? What impact do they have on the stock price?"
    result2 = rag_answer(
        question=question2,
        stock_code="TSLA.O",
        decision_time="2024-01-15T00:00:00",
        frequency="1d"
    )
    print(f"\nQuestion: {question2}")
    print(f"\nAnswer:\n{result2.answer}")
    print(f"\nVerification Status: {'Passed' if result2.passed else 'Failed'}")
    print(f"Mode: {result2.mode}")
    print()
    
    # Example 3: Trading history query
    print("Example 3: Trading History Query")
    print("-" * 80)
    question3 = "What is the recent trading history of MSFT.O?"
    result3 = rag_answer(
        question=question3,
        stock_code="MSFT.O",
        decision_time="2024-01-15T00:00:00",
        frequency="1d"
    )
    print(f"\nQuestion: {question3}")
    print(f"\nAnswer:\n{result3.answer}")
    print(f"\nVerification Status: {'Passed' if result3.passed else 'Failed'}")
    print(f"Mode: {result3.mode}")
    print()
    
    # Example 4: Risk check
    print("Example 4: Risk Check")
    print("-" * 80)
    question4 = "What are the current risks for NVDA.O?"
    result4 = rag_answer(
        question=question4,
        stock_code="NVDA.O",
        decision_time="2024-01-15T00:00:00",
        frequency="1d"
    )
    print(f"\nQuestion: {question4}")
    print(f"\nAnswer:\n{result4.answer}")
    print(f"\nVerification Status: {'Passed' if result4.passed else 'Failed'}")
    print(f"Mode: {result4.mode}")
    print()
    
    # Example 5: Trend analysis query
    print("Example 5: Trend Analysis Query")
    print("-" * 80)
    question5 = "Has AAPL.O been rising recently?"
    result5 = rag_answer(
        question=question5,
        stock_code="AAPL.O",
        decision_time="2024-01-15T00:00:00",
        frequency="1d"
    )
    print(f"\nQuestion: {question5}")
    print(f"\nAnswer:\n{result5.answer}")
    print(f"\nVerification Status: {'Passed' if result5.passed else 'Failed'}")
    print(f"Mode: {result5.mode}")
    if result5.violations:
        print(f"Violations: {result5.violations}")
    print()
    
    # Example 6: NVDA.O Trading History Query
    print("Example 6: NVDA.O Trading History Query")
    print("-" * 80)
    question6 = "What is the recent trading history of NVDA.O? Show me the buy and sell actions."
    result6 = rag_answer(
        question=question6,
        stock_code="NVDA.O",
        decision_time="2024-01-15T00:00:00",
        frequency="1d"
    )
    print(f"\nQuestion: {question6}")
    print(f"\nAnswer:\n{result6.answer}")
    print(f"\nVerification Status: {'Passed' if result6.passed else 'Failed'}")
    print(f"Mode: {result6.mode}")
    print()
    
    # Example 7: NVDA.O Feature Analysis
    print("Example 7: NVDA.O Feature Analysis")
    print("-" * 80)
    question7 = "What are the key technical features of NVDA.O? Analyze the return rates, volatility, and financial ratios."
    result7 = rag_answer(
        question=question7,
        stock_code="NVDA.O",
        decision_time="2024-01-15T00:00:00",
        frequency="1d"
    )
    print(f"\nQuestion: {question7}")
    print(f"\nAnswer:\n{result7.answer}")
    print(f"\nVerification Status: {'Passed' if result7.passed else 'Failed'}")
    print(f"Mode: {result7.mode}")
    print()
    
    # Example 8: NVDA.O News and Trading Correlation
    print("Example 8: NVDA.O News and Trading Correlation")
    print("-" * 80)
    question8 = "What news events happened around NVDA.O trading actions? Are there correlations between news sentiment and trading decisions?"
    result8 = rag_answer(
        question=question8,
        stock_code="NVDA.O",
        decision_time="2024-01-15T00:00:00",
        frequency="1d"
    )
    print(f"\nQuestion: {question8}")
    print(f"\nAnswer:\n{result8.answer}")
    print(f"\nVerification Status: {'Passed' if result8.passed else 'Failed'}")
    print(f"Mode: {result8.mode}")
    print()
    
    # Example 9: NVDA.O Market Trend and Volatility
    print("Example 9: NVDA.O Market Trend and Volatility")
    print("-" * 80)
    question9 = "What is the market trend and volatility pattern of NVDA.O in the past month? How does it compare to historical patterns?"
    result9 = rag_answer(
        question=question9,
        stock_code="NVDA.O",
        decision_time="2024-01-15T00:00:00",
        frequency="1d"
    )
    print(f"\nQuestion: {question9}")
    print(f"\nAnswer:\n{result9.answer}")
    print(f"\nVerification Status: {'Passed' if result9.passed else 'Failed'}")
    print(f"Mode: {result9.mode}")
    print()
    
    # Example 10: NVDA.O Comprehensive Decision Support
    print("Example 10: NVDA.O Comprehensive Decision Support")
    print("-" * 80)
    question10 = "Based on recent trading history, news events, and technical features, what is the current market situation for NVDA.O? Should I consider buying or selling?"
    result10 = rag_answer(
        question=question10,
        stock_code="NVDA.O",
        decision_time="2024-01-15T00:00:00",
        frequency="1d"
    )
    print(f"\nQuestion: {question10}")
    print(f"\nAnswer:\n{result10.answer}")
    print(f"\nVerification Status: {'Passed' if result10.passed else 'Failed'}")
    print(f"Mode: {result10.mode}")
    if result10.violations:
        print(f"Violations: {result10.violations}")
    print()
    
    print("=" * 80)
    print("Test Completed - All 10 Examples Executed")
    print("=" * 80)


if __name__ == "__main__":
    main()

