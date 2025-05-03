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
        super().__init__()
        self.queue = queue 
    
    async def on_llm_new_token(self, token: str, **kwargs: Any) -> None:
        """Process new tokens as they are generated."""
        await self.queue.put(token)
        
    async def aiter(self):
        """Async iterator for tokens."""
        while True:
            try:
                token = await asyncio.wait_for(self.queue.get(), timeout=30.0)
                yield token
                self.queue.task_done()
            except asyncio.TimeoutError:
                # If no tokens for 30 seconds, end streaming
                break
            except asyncio.CancelledError:
                # Handle cancellation
                break
        
    def on_llm_new_token_sync(self, token: str, **kwargs: Any) -> None:
        """Synchronous fallback for token handling - needed for some LangChain versions."""
        # Use asyncio.run_coroutine_threadsafe or create_task in the running loop if available
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                asyncio.create_task(self.queue.put(token))
            else:
                # Fallback in case we're not in an async context
                asyncio.run(self.queue.put(token))
        except RuntimeError:
            # If we can't get event loop, just print the token directly
            print(token, end="", flush=True)

class StreamingStdOutCallbackHandler(BaseCallbackHandler): 
    """Handler for streaming tokens to standard output."""

    def on_llm_new_token(self, token: str, **kwargs: Any) -> None:
        """Print tokens to stdout as they're generated."""
        print(token, end="", flush=True)

        