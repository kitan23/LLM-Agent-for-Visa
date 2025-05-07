"""
Direct Streaming Module for LLM Generation

This module provides the DirectStreamer class for handling token streaming 
directly from transformer models.
"""

import asyncio
import threading
import logging
import time
from typing import AsyncIterator, Optional, List, Any
import torch
from transformers import TextIteratorStreamer

logger = logging.getLogger("opt_rag.direct_streamer")

class DirectStreamer:
    """Class for direct streaming of tokens from a transformer model."""
    
    def __init__(self, model, tokenizer, device):
        """Initialize the DirectStreamer.
        
        Args:
            model: The transformer model
            tokenizer: The tokenizer corresponding to the model
            device: The device to run the model on (cuda, mps, cpu)
        """
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.skip_special_tokens = True
    
    async def generate_and_stream(
        self, 
        prompt: str, 
        max_tokens: int = 1024,
        temperature: float = 0.7,
        top_p: float = 0.9,
        repetition_penalty: float = 1.1,
        cancel_event: Optional[threading.Event] = None
    ) -> AsyncIterator[str]:
        """Generate and stream tokens for a given prompt.
        
        Args:
            prompt: The input prompt to generate from
            max_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature
            top_p: Top-p sampling parameter
            repetition_penalty: Penalty for repetition
            cancel_event: Event that can be set to cancel generation
            
        Yields:
            Tokens as they are generated
        """
        logger.info("Setting up streamer")
        
        # Check for early cancellation
        if cancel_event and cancel_event.is_set():
            logger.info("Cancellation requested before streamer setup")
            return
        
        # Format messages for model
        messages = [{"role": "user", "content": prompt}]
        
        # Set up streamer for token streaming
        streamer = TextIteratorStreamer(
            self.tokenizer, 
            skip_prompt=True,
            skip_special_tokens=self.skip_special_tokens
        )
        
        # Check if cancel event was set before starting
        if cancel_event and cancel_event.is_set():
            logger.info("Cancellation requested before generation started")
            return
        
        # Start generation in a separate thread
        logger.info("Setting up generation thread")
        inputs = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt"
        ).to(self.device)
        
        generation_kwargs = dict(
            inputs=inputs,
            max_new_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            do_sample=True,
            streamer=streamer
        )
        
        thread = threading.Thread(
            target=self._generate_with_cancel,
            args=(generation_kwargs, cancel_event)
        )
        thread.start()
        
        logger.info("Started generation thread, waiting for tokens")
        
        # Stream tokens as they're generated
        start_time = time.time()
        tokens_streamed = 0
        
        # Set up buffer to handle multi-byte characters
        buffer = ""
        
        try:
            # Stream tokens with rate limiting
            for token in streamer:
                # Check for cancellation
                if cancel_event and cancel_event.is_set():
                    logger.info("Streaming cancelled during generation")
                    break
                    
                buffer += token
                
                # For multi-byte characters, we need to ensure we're yielding complete UTF-8 sequences
                # This is especially important for languages with non-ASCII characters
                try:
                    # Try to encode - if it works, we have a complete character
                    buffer.encode('utf-8')
                    
                    # Yield the buffer and reset
                    yield buffer
                    buffer = ""
                    tokens_streamed += 1
                    
                    # Log progress periodically
                    if tokens_streamed % 20 == 0:
                        elapsed = time.time() - start_time
                        rate = tokens_streamed / elapsed if elapsed > 0 else 0
                        logger.info(f"Streamed {tokens_streamed} tokens ({rate:.1f} tokens/sec)")
                    
                    # Rate limiting to avoid overwhelming the client
                    # This small sleep helps ensure the frontend can keep up
                    if tokens_streamed % 5 == 0:
                        await asyncio.sleep(0.01)
                        
                except UnicodeEncodeError:
                    # If encoding fails, we have an incomplete character
                    # Keep in buffer until we get more tokens to complete it
                    continue
                    
            # Yield any remaining text in the buffer
            if buffer and not (cancel_event and cancel_event.is_set()):
                yield buffer
                
        except Exception as e:
            logger.error(f"Error during streaming: {e}", exc_info=True)
            if not (cancel_event and cancel_event.is_set()):
                yield f"Error during streaming: {str(e)}"
        finally:
            generation_done = time.time() - start_time
            logger.info(f"Streaming complete, streamed {tokens_streamed} tokens in {generation_done:.2f}s")
            
            # Wait for the generation thread to finish (timeout after 2s if it's hanging)
            thread.join(timeout=2.0)
            if thread.is_alive():
                logger.warning("Generation thread did not terminate within timeout")
        
    def _generate_with_cancel(self, generation_kwargs: dict, cancel_event: Optional[threading.Event] = None):
        """Run the model generation with cancel support.
        
        Args:
            generation_kwargs: Keyword arguments for model.generate
            cancel_event: Event that can be set to cancel generation
        """
        try:
            # Check for cancellation before starting
            if cancel_event and cancel_event.is_set():
                logger.info("Generation cancelled before starting")
                return
                
            # Start generation
            self.model.generate(**generation_kwargs)
            
        except Exception as e:
            logger.error(f"Error during generation: {e}", exc_info=True)
            
        finally:
            # If we're cancelled during generation, try to clean up resources
            if cancel_event and cancel_event.is_set():
                # Clean up anything we need to
                logger.info("Generation loop terminated due to cancellation") 