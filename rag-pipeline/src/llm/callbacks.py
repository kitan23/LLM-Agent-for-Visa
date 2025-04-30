"""
Callback handlers for LLM response streaming.

This module provides custom callback handlers for streaming LLM responses
in both synchronous and asynchronous contexts.
"""


import asyncio 
from typing import Any, Dict, List, Optional 
from langchain.callbacks.base import BaseCallbackHandler 

class StreamingCallbackHandler(BaseCallbackHandler): 
    """
    Custom callback handler for streaming responses.
    """

    def __init__(self):
        self.tokens = []
        self.text = ""

    def on_llm_new_token(self, token: str, **kwargs: Any) -> None:
        """Process new tokens as they are generated."""
        self.tokens.append(token)
        self.text += token

    def get_tokens(self) -> list:
        """Return collected tokens."""
        return self.tokens

    def get_text(self) -> str:
        """Return collected text."""
        return self.text
    


class AsyncStreamingCallbackHandler(BaseCallbackHandler): 
    """
    Custom callback handler for asynchronous streaming responses.
    """

    def __init__(self, queue: asyncio.Queue):
        """Initialize the handler with an async queue.
        Args:
            queue: An asyncio queue for collecting tokens and text.
        
        """
        self.queue = queue 
    
    async def on_llm_new_token(self, token: str, **kwargs: Any) -> None:
        """Process new tokens as they are generated."""
        await self.queue.put(token)

class StreamingStdOutCallbackHandler(BaseCallbackHandler): 
    """Handler for streaming tokens to standard output."""

    def on_llm_new_token(self, token: str, **kwargs: Any) -> None:
        """Print tokens to stdout as they're generated."""
        print(token, end="", flush=True)

        