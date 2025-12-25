"""
Generator
Call LLM to generate answer
"""
import sys
from pathlib import Path
import re

project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from trader.config import get_deepseek_api_key
from trader.rag import get_rag_logger

logger = get_rag_logger(__name__)


def llm_generate(prompt: str, model: str = "deepseek-chat", temperature: float = 0.3) -> str:
    """
    Call LLM to generate answer
    
    Args:
        prompt: Prompt
        model: Model name
        temperature: Temperature parameter
        
    Returns:
        Raw answer string
    """
    try:
        from openai import OpenAI
    except ImportError:
        logger.error("openai module not installed, cannot generate LLM. Please run: pip install openai")
        raise
    
    try:
        # Initialize DeepSeek API client
        api_key = get_deepseek_api_key()
        client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")
        
        # Parse system and user parts
        system_match = re.search(r'<system>(.*?)</system>', prompt, re.DOTALL)
        user_match = re.search(r'<user>(.*?)</user>', prompt, re.DOTALL)
        
        messages = []
        if system_match:
            messages.append({"role": "system", "content": system_match.group(1).strip()})
        if user_match:
            messages.append({"role": "user", "content": user_match.group(1).strip()})
        
        if not messages:
            # If no tags found, entire prompt as user message
            messages = [{"role": "user", "content": prompt}]
        
        logger.info(f"Calling LLM API (model={model})...")
        
        # Call API
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            stream=False,
            temperature=temperature
        )
        
        raw_answer = response.choices[0].message.content
        logger.info("LLM generation completed")
        
        return raw_answer
        
    except Exception as e:
        logger.error(f"LLM generation failed: {e}", exc_info=True)
        raise

