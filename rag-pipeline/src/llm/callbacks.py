"""
Callback handlers for LLM response streaming.

This module provides custom callback handlers for streaming LLM responses
in both synchronous and asynchronous contexts.
"""


import asyncio 
import logging
from typing import Any, Dict, List, Optional 
from langchain.callbacks.base import BaseCallbackHandler 

logger = logging.getLogger("opt_rag.callbacks")

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
        logger.info("AsyncStreamingCallbackHandler initialized with queue")
    
    async def on_llm_new_token(self, token: str, **kwargs: Any) -> None:
        """Process new tokens as they are generated."""
        logger.debug(f"New token received: {token!r}")
        await self.queue.put(token)
        logger.debug(f"Token added to queue, current size: {self.queue.qsize()}")
        
    async def aiter(self):
        """Async iterator for tokens."""
        logger.info("Starting aiter for token streaming")
        while True:
            try:
                logger.debug("Waiting for token from queue...")
                token = await asyncio.wait_for(self.queue.get(), timeout=60.0)  # Increased timeout
                logger.debug(f"Got token from queue: {token!r}")
                
                # Empty token signals end of generation
                if token == "":
                    logger.info("Received empty token, ending stream")
                    break
                    
                yield token
                self.queue.task_done()
            except asyncio.TimeoutError:
                # If no tokens for 60 seconds, end streaming
                logger.warning("Timeout waiting for tokens, ending stream")
                break
            except asyncio.CancelledError:
                # Handle cancellation
                logger.warning("Token stream was cancelled")
                break
            except Exception as e:
                logger.error(f"Error in token streaming: {e}")
                break
        
        logger.info("Token streaming complete")
        
    def on_llm_new_token_sync(self, token: str, **kwargs: Any) -> None:
        """Synchronous fallback for token handling - needed for some LangChain versions."""
        # Use asyncio.run_coroutine_threadsafe or create_task in the running loop if available
        logger.debug(f"Sync token received: {token!r}")
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                asyncio.create_task(self.queue.put(token))
                logger.debug("Token added to queue via create_task")
            else:
                # Fallback in case we're not in an async context
                asyncio.run(self.queue.put(token))
                logger.debug("Token added to queue via asyncio.run")
        except RuntimeError as e:
            # If we can't get event loop, just print the token directly
            logger.error(f"Error adding token to queue: {e}")
            print(token, end="", flush=True)

class StreamingStdOutCallbackHandler(BaseCallbackHandler): 
    """Handler for streaming tokens to standard output."""

    def on_llm_new_token(self, token: str, **kwargs: Any) -> None:
        """Print tokens to stdout as they're generated."""
        print(token, end="", flush=True)

        