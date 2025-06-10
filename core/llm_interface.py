# c:\\WITS\\WitsV3\\core\\llm_interface.py
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, AsyncGenerator, List, TypeVar, Callable, Awaitable
import httpx
import json
import os
from unittest.mock import AsyncMock  # Add this import for test compatibility
import logging
import asyncio

# Import AppConfig and relevant settings models from core.config
from .config import WitsV3Config, OllamaSettings

logger = logging.getLogger(__name__)

# Define a generic type for the retry function
T = TypeVar('T')

class BaseLLMInterface(ABC):
    def __init__(self, config: WitsV3Config):
        self.config = config

    @abstractmethod
    async def generate_text(
        self,
        prompt: str,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        stop_sequences: Optional[List[str]] = None,
    ) -> str:
        """Generate text from the LLM."""
        pass

    @abstractmethod
    async def stream_text(
        self,
        prompt: str,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        stop_sequences: Optional[List[str]] = None,
    ) -> AsyncGenerator[str, None]:
        """Stream text from the LLM."""
        pass

    @abstractmethod
    async def get_embedding(
        self,
        text: str,
        model: Optional[str] = None
    ) -> List[float]:
        """Get an embedding for the given text."""
        pass

class OllamaInterface(BaseLLMInterface):
    def __init__(self, config: WitsV3Config):
        super().__init__(config)
        self.ollama_settings: OllamaSettings = config.ollama_settings
        # TODO: Consider managing the lifecycle of the client, e.g., closing it on app shutdown.
        self.http_client = httpx.AsyncClient(timeout=self.ollama_settings.request_timeout)
        self.logger = logging.getLogger("WitsV3.OllamaInterface")

    async def _prepare_payload(
        self,
        prompt: str,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        stop_sequences: Optional[List[str]] = None,
        stream: bool = False
    ) -> Dict[str, Any]:
        effective_model = model or self.ollama_settings.default_model
        options: Dict[str, Any] = {
            "temperature": temperature if temperature is not None else self.config.agents.default_temperature
        }
        if max_tokens is not None:
            options["num_predict"] = max_tokens
        if stop_sequences is not None:
            options["stop"] = stop_sequences

        return {
            "model": effective_model,
            "prompt": prompt,
            "stream": stream,
            "options": options
        }

    async def _execute_with_retry(self, operation_name: str, func: Callable[..., Awaitable[T]], *args, **kwargs) -> T:
        """Execute a function with retry logic on failure."""
        max_attempts = self.ollama_settings.retry_attempts
        base_delay = self.ollama_settings.retry_delay
        use_exponential_backoff = self.ollama_settings.exponential_backoff

        for attempt in range(1, max_attempts + 1):
            try:
                return await func(*args, **kwargs)
            except httpx.HTTPStatusError as e:
                error_msg = f"{operation_name} failed (Attempt {attempt}/{max_attempts}): HTTP {e.response.status_code}"

                # Service unavailable (503) or too many requests (429) - retry
                if e.response.status_code in (429, 503, 502, 504) and attempt < max_attempts:
                    # Get retry delay - use exponential backoff if configured
                    delay = base_delay * (2 ** (attempt - 1)) if use_exponential_backoff else base_delay

                    self.logger.warning(f"{error_msg}. Retrying in {delay:.2f} seconds...")
                    await asyncio.sleep(delay)
                    continue

                # Process specific error codes
                if e.response.status_code == 404:
                    self.logger.error(f"{error_msg}. Model not found.")
                    raise ValueError(f"Model not found. Check if the model is available in Ollama.") from e
                elif e.response.status_code == 400:
                    self.logger.error(f"{error_msg}. Bad request: {e.response.text}")
                    raise ValueError(f"Bad request to Ollama API: {e.response.text}") from e
                elif e.response.status_code == 401:
                    self.logger.error(f"{error_msg}. Authentication required.")
                    raise ValueError("Authentication required for Ollama API") from e

                # For other status codes
                self.logger.error(f"{error_msg}. Response: {e.response.text}")
                raise

            except httpx.RequestError as e:
                error_msg = f"{operation_name} failed (Attempt {attempt}/{max_attempts}): Connection error"

                if attempt < max_attempts:
                    delay = base_delay * (2 ** (attempt - 1)) if use_exponential_backoff else base_delay
                    self.logger.warning(f"{error_msg}. Retrying in {delay:.2f} seconds...")
                    await asyncio.sleep(delay)
                    continue

                self.logger.error(f"{error_msg}. Ollama service may be unavailable at {self.ollama_settings.url}.")
                raise ValueError(f"Failed to connect to Ollama at {self.ollama_settings.url}. Please ensure Ollama is running.") from e

            except Exception as e:
                self.logger.error(f"{operation_name} failed with unexpected error: {str(e)}")
                raise

        # This should never be reached due to the for loop structure and exception handling
        raise RuntimeError(f"{operation_name} failed after {max_attempts} attempts")

    async def generate_text(
        self,
        prompt: str,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        stop_sequences: Optional[List[str]] = None,
    ) -> str:
        payload = await self._prepare_payload(prompt, model, temperature, max_tokens, stop_sequences, stream=False)

        async def _generate() -> str:
            response = await self.http_client.post(
                f"{self.ollama_settings.url}/api/generate",
                json=payload,
            )
            response.raise_for_status()
            response_data = response.json()
            return response_data.get("response", "")

        return await self._execute_with_retry("Text generation", _generate)

    async def stream_text(
        self,
        prompt: str,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        stop_sequences: Optional[List[str]] = None,
    ) -> AsyncGenerator[str, None]:
        """
        Stream text from the LLM.

        Args:
            prompt: The prompt to send to the LLM
            model: Optional model name (defaults to config)
            temperature: Optional temperature (defaults to config)
            max_tokens: Optional max tokens (defaults to config)
            stop_sequences: Optional stop sequences

        Yields:
            Text chunks from the LLM
        """
        payload = await self._prepare_payload(prompt, model, temperature, max_tokens, stop_sequences, stream=True)
        url = f"{self.ollama_settings.url}/api/generate"

        # Initialize counters for retry logic
        attempt = 0
        max_attempts = self.ollama_settings.retry_attempts
        base_delay = self.ollama_settings.retry_delay
        use_exponential_backoff = self.ollama_settings.exponential_backoff

        while attempt < max_attempts:
            attempt += 1
            try:
                # This properly handles streaming in httpx
                async with self.http_client.stream(
                    "POST",
                    url,
                    json=payload,
                    timeout=self.ollama_settings.request_timeout
                ) as response:
                    response.raise_for_status()
                    # Process streaming response line by line
                    async for line in response.aiter_lines():
                        if not line:
                            continue

                        try:
                            chunk = json.loads(line)
                            if "response" in chunk and not chunk.get("done", False):
                                yield chunk["response"]
                            elif chunk.get("done"):
                                # End of stream
                                break
                        except json.JSONDecodeError:
                            self.logger.warning(f"Failed to decode JSON stream line: {line}")

                # If we reach here, streaming completed successfully
                break

            except httpx.HTTPStatusError as e:
                error_msg = f"Stream generation failed (Attempt {attempt}/{max_attempts}): HTTP {e.response.status_code}"

                # Only retry for certain status codes and if we have attempts left
                if e.response.status_code in (429, 503, 502, 504) and attempt < max_attempts:
                    delay = base_delay * (2 ** (attempt - 1)) if use_exponential_backoff else base_delay
                    self.logger.warning(f"{error_msg}. Retrying in {delay:.2f} seconds...")
                    await asyncio.sleep(delay)
                    continue

                # Handle specific error codes
                if e.response.status_code == 404:
                    self.logger.error(f"{error_msg}. Model not found.")
                    yield f"\n\nError: Model not found. Please check if the model is available in Ollama.\n"
                elif e.response.status_code == 400:
                    self.logger.error(f"{error_msg}. Bad request: {e.response.text}")
                    yield f"\n\nError: Invalid request to Ollama API: {e.response.text}\n"
                else:
                    self.logger.error(f"{error_msg}. Response: {e.response.text}")
                    yield f"\n\nError: Ollama API error ({e.response.status_code}). Please check logs for details.\n"
                break

            except httpx.RequestError as e:
                error_msg = f"Stream generation failed (Attempt {attempt}/{max_attempts}): Connection error"

                if attempt < max_attempts:
                    delay = base_delay * (2 ** (attempt - 1)) if use_exponential_backoff else base_delay
                    self.logger.warning(f"{error_msg}. Retrying in {delay:.2f} seconds...")
                    await asyncio.sleep(delay)
                    continue

                self.logger.error(f"{error_msg}. Ollama service may be unavailable at {self.ollama_settings.url}.")
                yield f"\n\nError: Failed to connect to Ollama at {self.ollama_settings.url}. Please ensure Ollama is running.\n"
                break

            except Exception as e:
                self.logger.error(f"Stream generation failed with unexpected error: {str(e)}")
                yield f"\n\nError: An unexpected error occurred. Please check logs for details.\n"
                break

    async def get_embedding(
        self,
        text: str,
        model: Optional[str] = None
    ) -> List[float]:
        effective_model = model or self.ollama_settings.embedding_model
        payload = {
            "model": effective_model,
            "prompt": text
        }

        async def _get_embedding() -> List[float]:
            response = await self.http_client.post(
                f"{self.ollama_settings.url}/api/embeddings",
                json=payload,
            )
            response.raise_for_status()
            embedding = response.json().get("embedding", [])

            # Check if the embedding is not empty
            if not embedding:
                self.logger.warning("Received empty embedding from Ollama API")

            return embedding

        return await self._execute_with_retry("Embedding generation", _get_embedding)

    async def is_service_available(self) -> bool:
        """Check if the Ollama service is available."""
        try:
            response = await self.http_client.get(
                f"{self.ollama_settings.url}/api/tags",
                timeout=5.0  # Short timeout for availability check
            )
            response.raise_for_status()
            return True
        except Exception as e:
            self.logger.warning(f"Ollama service check failed: {str(e)}")
            return False

    async def shutdown(self) -> None:
        """Close the HTTP client."""
        if self.http_client:
            await self.http_client.aclose()
            self.logger.info("Ollama interface HTTP client closed")

def get_llm_interface(config: WitsV3Config) -> BaseLLMInterface:
    if config.llm_interface.default_provider == "adaptive":
        # Import here to avoid circular imports
        from .adaptive_llm_interface import AdaptiveLLMInterface

        # First create a base LLM interface (Ollama by default)
        base_llm = OllamaInterface(config)

        # Then create the adaptive LLM interface with the base LLM
        return AdaptiveLLMInterface(config, base_llm)
    elif config.llm_interface.default_provider == "ollama":
        return OllamaInterface(config)
    # Example for future extension:
    # elif config.llm_interface.default_provider == "openai":
    #     return OpenAIInterface(config) # To be implemented
    else:
        raise ValueError(f"Unsupported LLM provider: {config.llm_interface.default_provider}")

# Example usage (for testing this file directly)
if __name__ == "__main__":
    import asyncio
    from .config import WitsV3Config # Relative import for testing

    async def main_test():
        # Construct the path to config.yaml relative to this script's location
        current_script_dir = os.path.dirname(os.path.abspath(__file__))
        project_root_dir = os.path.dirname(current_script_dir) # Moves from core to WitsV3
        config_file_path = os.path.join(project_root_dir, "config.yaml")

        print(f"Attempting to load configuration from: {config_file_path}")
        app_config = WitsV3Config.from_yaml(config_file_path)

        if not app_config:
            print("Failed to load configuration. Exiting test.")
            return

        print("Configuration loaded. Initializing LLM interface...")
        try:
            llm_interface = get_llm_interface(app_config)
        except Exception as e:
            print(f"Error initializing LLM interface: {e}")
            return

        print(f"Using Ollama URL: {app_config.ollama_settings.url}")
        print(f"Default model for generation: {app_config.ollama_settings.default_model}")
        print(f"Embedding model: {app_config.ollama_settings.embedding_model}")

        try:
            # Check if service is available
            if isinstance(llm_interface, OllamaInterface):
                available = await llm_interface.is_service_available()
                if not available:
                    print("\n⚠️ WARNING: Ollama service appears to be unavailable.")
                    print(f"Please ensure Ollama is running at {app_config.ollama_settings.url}\n")

            # Test non-streaming generation
            print("\n--- Non-Streaming Generation Test ---")
            response = await llm_interface.generate_text(
                prompt="Why is the sky blue? Explain briefly.",
                model=app_config.ollama_settings.default_model # Explicitly pass model for clarity
            )
            print(f"Full Response:\n{response}")

            # Test streaming generation
            print("\n--- Streaming Generation Test ---")
            print("Streamed Response:")
            async for chunk in llm_interface.stream_text(
                prompt="Tell me a very short story about a curious robot.",
                model=app_config.ollama_settings.default_model
            ):
                print(chunk, end="", flush=True)
            print("\n--- End of Stream ---")

            # Test embedding generation
            print("\n--- Embedding Generation Test ---")
            embedding = await llm_interface.get_embedding(
                text="Hello WitsV3!",
                model=app_config.ollama_settings.embedding_model
            )
            if embedding:
                print(f"Generated embedding with {len(embedding)} dimensions.")
                print(f"First 5 dimensions: {embedding[:5]}")
            else:
                print("Failed to generate embedding or embedding was empty.")

        except ValueError as e:
            print(f"\n⚠️ {e}")
        except httpx.RequestError:
            print(f"\n⚠️ Connection Error: Could not connect to Ollama at {app_config.ollama_settings.url}.")
            print("Please ensure Ollama is running and accessible.")
        except Exception as e:
            print(f"\n⚠️ An unexpected error occurred during LLM interface tests: {e}")
        finally:
            # Cleanup
            if isinstance(llm_interface, OllamaInterface):
                await llm_interface.shutdown()

    asyncio.run(main_test())
