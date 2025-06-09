# c:\\WITS\\WitsV3\\core\\llm_interface.py
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, AsyncGenerator, List
import httpx
import json
import os
from unittest.mock import AsyncMock  # Add this import for test compatibility

# Import AppConfig and relevant settings models from core.config
from .config import WitsV3Config, OllamaSettings

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
        pass
        # This yield is necessary for Python to recognize it as an async generator
        # even if the abstract method itself doesn't yield. Implementations will.
        if False:
            yield

    @abstractmethod
    async def get_embedding(
        self,
        text: str,
        model: Optional[str] = None
    ) -> List[float]:
        pass

class OllamaInterface(BaseLLMInterface):
    def __init__(self, config: WitsV3Config):
        super().__init__(config)
        self.ollama_settings: OllamaSettings = config.ollama_settings
        # TODO: Consider managing the lifecycle of the client, e.g., closing it on app shutdown.
        self.http_client = httpx.AsyncClient(timeout=self.ollama_settings.request_timeout)

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

    async def generate_text(
        self,
        prompt: str,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        stop_sequences: Optional[List[str]] = None,
    ) -> str:
        payload = await self._prepare_payload(prompt, model, temperature, max_tokens, stop_sequences, stream=False)
        try:
            response = await self.http_client.post(
                f"{self.ollama_settings.url}/api/generate",
                json=payload,
            )
            response.raise_for_status()
            response_data = response.json()
            return response_data.get("response", "")
        except httpx.HTTPStatusError as e:
            print(f"Ollama API request failed: {e.response.status_code} - {e.response.text}")
            # Consider raising a custom, more specific error
            raise
        except httpx.RequestError as e:
            print(f"Ollama request failed: {e}")
            raise

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

        # Keep a reference to the stream and response to prevent it from being garbage collected
        stream_response = None

        try:
            # For real use with Ollama, use streaming=True
            url = f"{self.ollama_settings.url}/api/generate"

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
                        print(f"Warning: Failed to decode JSON stream line: {line}")

        except httpx.HTTPStatusError as e:
            print(f"Ollama API stream request failed: {e.response.status_code} - {e.response.text}")
            raise
        except httpx.RequestError as e:
            print(f"Ollama stream request failed: {e}")
            raise

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
        try:
            response = await self.http_client.post(
                f"{self.ollama_settings.url}/api/embeddings",
                json=payload,
            )
            response.raise_for_status()
            return response.json().get("embedding", [])
        except httpx.HTTPStatusError as e:
            print(f"Ollama embedding request failed: {e.response.status_code} - {e.response.text}")
            raise
        except httpx.RequestError as e:
            print(f"Ollama embedding request failed: {e}")
            raise

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
            return        print(f"Using Ollama URL: {app_config.ollama_settings.url}")
        print(f"Default model for generation: {app_config.ollama_settings.default_model}")
        print(f"Embedding model: {app_config.ollama_settings.embedding_model}")

        try:
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

        except httpx.RequestError as e:
            print(f"\nConnection Error: Could not connect to Ollama at {app_config.ollama_settings.url}.")
            print("Please ensure Ollama is running and accessible.")
            print(f"Details: {e}")
        except Exception as e:
            print(f"\nAn unexpected error occurred during LLM interface tests: {e}")

    asyncio.run(main_test())
