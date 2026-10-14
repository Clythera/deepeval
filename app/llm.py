"""LLM client for communicating with OpenAI-compatible LLMs."""

import os
from typing import Optional
from openai import AsyncOpenAI, OpenAIError
from dotenv import load_dotenv
import asyncio
from app.prompts import SYSTEM_PROMPT

# Load environment variables
load_dotenv()

# Configuration from environment variables
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "http://localhost:11434/v1")
LLM_API_KEY = os.getenv("LLM_API_KEY", "not-needed")
LLM_MODEL = os.getenv("LLM_MODEL", "qwen3-coder:30b")
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.2"))
LLM_MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", "1000"))
DEFAULT_TIMEOUT_SECONDS = 90

# Initialize the OpenAI client
client = AsyncOpenAI(
    base_url=LLM_BASE_URL,
    api_key=LLM_API_KEY,
)


async def ask_llm(question: str, timeout_seconds: int = DEFAULT_TIMEOUT_SECONDS) -> str:
    """
    Send a question to the LLM and return the generated response.
    
    Args:
        question: The question to ask the LLM
        timeout_seconds: Timeout in seconds for the LLM request
    
    Returns:
        The generated response from the LLM
    
    Raises:
        OpenAIError: If there's an error communicating with the LLM
        asyncio.TimeoutError: If the request times out
    """
    try:
        # Create a timeout context for the LLM call
        response = await asyncio.wait_for(
            client.chat.completions.create(
                model=LLM_MODEL,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": question}
                ],
                temperature=LLM_TEMPERATURE,
                max_tokens=LLM_MAX_TOKENS,
            ),
            timeout=timeout_seconds
        )
        
        # Extract and return the response content
        if response.choices and len(response.choices) > 0:
            return response.choices[0].message.content.strip()
        else:
            raise OpenAIError("No response received from LLM")
            
    except asyncio.TimeoutError:
        raise asyncio.TimeoutError(f"LLM request timed out after {timeout_seconds} seconds")
    except OpenAIError as e:
        # Re-raise OpenAI errors but don't expose sensitive information
        raise OpenAIError(f"Failed to get response from LLM: {str(e)}")
    except Exception as e:
        # Handle any other unexpected errors
        raise OpenAIError(f"Unexpected error communicating with LLM: {str(e)}")