"""
Provider module.
Abstracts the underlying LLM - Anthropic Claude, local Ollama, or llama.cpp.

The Oracle communicates through a rigid type system - schema validation
with error feedback. This makes "tool calling" model-agnostic: any model
that can output JSON and learn from feedback works with our system.
"""

import logging
import os
import requests

logger = logging.getLogger(__name__)
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Dict, Generator, List, Optional, Tuple, Union


class ThinkLevel(Enum):
    """Abstract thinking intensity - provider maps to model-specific values."""
    OFF = "off"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


@dataclass(frozen=True)
class ProviderResponse:
    """
    Immutable result from a provider completion.

    FP pattern: return all data instead of storing on object.
    No duck punching, no hidden state, full type safety.
    """
    text: str
    thinking: str = ""
    done_reason: str = ""  # "stop", "length", etc.
    eval_count: int = 0


class Provider(ABC):
    """Base class for LLM providers."""

    @abstractmethod
    def complete(self, messages: List[Dict[str, str]], max_tokens: int = 300, model: Optional[str] = None, seed: Optional[int] = None, temperature: Optional[float] = None, think_level: Optional[ThinkLevel] = None) -> ProviderResponse:
        """
        Generate a completion from a list of messages.

        Args:
            messages: List of {"role": "user/assistant/system", "content": "..."}
            max_tokens: Maximum tokens to generate
            model: Optional model override (e.g., vision model when images present)
            seed: Optional random seed for reproducibility (or to perturb retries)
            temperature: Optional temperature override (lower = more deterministic)
            think_level: Optional thinking intensity (provider-specific)

        Returns:
            ProviderResponse with text, thinking, done_reason, eval_count
        """
        pass

    def stream(self, messages: List[Dict[str, str]], max_tokens: int = 300, model: Optional[str] = None) -> Generator[str, None, None]:
        """
        Stream a completion token by token.

        Default implementation falls back to complete() and yields the whole response.
        Override in subclasses for true streaming.

        Args:
            model: Optional model override (e.g., vision model when images present)

        Yields:
            Text chunks as they're generated
        """
        yield self.complete(messages, max_tokens, model=model).text

    @property
    def supports_streaming(self) -> bool:
        """Whether this provider supports true streaming."""
        return False

    @property
    @abstractmethod
    def name(self) -> str:
        """Provider name for display."""
        pass


class AnthropicProvider(Provider):
    """Anthropic Claude provider."""

    # Model name mapping: friendly name → full model ID
    MODELS = {
        "sonnet": "claude-sonnet-4-6",                # $3/$15 - default
        "opus": "claude-opus-4-6",                    # $5/$25 - frontier
    }

    # Maximum output tokens per model (from API error messages)
    MAX_OUTPUT = {
        "claude-sonnet-4-6": 64000,
        "claude-opus-4-6": 128000,
    }

    def __init__(self, api_key: str = None, model: str = "claude-sonnet-4-6"):
        import anthropic

        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        if not self.api_key:
            raise ValueError("Anthropic API key required")
        self.client = anthropic.Anthropic(api_key=self.api_key)
        # Resolve friendly name to full model ID
        self.model = self.MODELS.get(model, model)

    @property
    def name(self) -> str:
        return f"anthropic/{self.model}"

    def complete(self, messages: List[Dict[str, str]], max_tokens: int = 300, model: Optional[str] = None, seed: Optional[int] = None, temperature: Optional[float] = None, think_level: Optional[ThinkLevel] = None) -> ProviderResponse:
        """Generate completion using Anthropic API.

        Args:
            model: Optional model override (ignored for Anthropic - uses instance model)
            seed: Ignored - Anthropic doesn't support seed parameter
            temperature: Optional temperature override (default: 1.0 for Anthropic)
        """
        try:
            # Anthropic API doesn't support system role in messages array
            # Extract system message and pass separately
            system_prompt = None
            api_messages = []
            for msg in messages:
                if msg["role"] == "system":
                    system_prompt = msg["content"]
                else:
                    api_messages.append(msg)

            # Clamp to model's maximum output tokens
            model_max = self.MAX_OUTPUT.get(self.model, 16384)
            effective_max = min(max_tokens, model_max) if max_tokens > 0 else model_max

            kwargs = {
                "model": self.model,
                "max_tokens": effective_max,
                "messages": api_messages,
            }
            if system_prompt:
                kwargs["system"] = system_prompt
            if temperature is not None:
                kwargs["temperature"] = temperature

            # Stream and collect -- required by SDK for large max_tokens
            chunks = []
            with self.client.messages.stream(**kwargs) as stream:
                for text in stream.text_stream:
                    chunks.append(text)
            result = "".join(chunks).strip()
            return ProviderResponse(
                text=result,
                done_reason=stream.get_final_message().stop_reason or "stop",
            )
        except Exception as e:
            logger.error("Anthropic API error: %s", e)
            raise


class OllamaProvider(Provider):
    """
    Ollama local LLM provider.

    Connects to a local or remote Ollama instance. No native tool calling
    needed - our schema validation feedback loop handles structured output.

    Configuration:
        OLLAMA_HOST - Server address (default: http://localhost:11434)
        OLLAMA_MODEL - Model to use (default: nemotron-3-nano)

    Example:
        export OLLAMA_HOST=http://10.106.1.182:11434
        export OLLAMA_MODEL=nemotron-3-nano
    """

    # Context lengths for common models
    MODEL_CONTEXT_LENGTHS = {
        "nemotron": 32768,
        "qwen3": 128000,
        "qwen2.5": 128000,
        "llama3.3": 128000,
        "llama3.2": 128000,
        "mistral": 32768,
        "deepseek-r1": 64000,
        "phi4": 16384,
    }

    def __init__(
        self,
        model: Optional[str] = None,
        base_url: Optional[str] = None,
        temperature: float = 0.7,
        timeout: Optional[int] = None,  # None = no timeout (wait for completion)
        server_name: Optional[str] = None,  # e.g., "local", "big"
    ):
        self.base_url = (base_url or os.getenv("OLLAMA_HOST", "http://localhost:11434")).rstrip("/")
        self.model = model or os.getenv("OLLAMA_MODEL", "nemotron-3-nano")
        self.temperature = temperature
        self.timeout = timeout
        self.server_name = server_name

        # Determine context length from model name
        self._context_length = 8192  # Default
        for family, length in self.MODEL_CONTEXT_LENGTHS.items():
            if family in self.model.lower():
                self._context_length = length
                break

        # Store last thinking for inspection (Qwen3, etc.)
        self._last_thinking = ""
        self._last_content = ""
        self._last_done_reason = ""
        self._last_eval_count = 0

    @property
    def name(self) -> str:
        # Include server name to distinguish same model on different servers
        if self.server_name:
            return f"ollama/{self.model}@{self.server_name}"
        return f"ollama/{self.model}"

    @property
    def context_length(self) -> int:
        return self._context_length

    @property
    def last_thinking(self) -> str:
        return self._last_thinking

    @property
    def last_content(self) -> str:
        return self._last_content

    @property
    def last_done_reason(self) -> str:
        return self._last_done_reason

    @property
    def last_eval_count(self) -> int:
        return self._last_eval_count

    def _build_payload(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
        seed: Optional[int] = None,
        temperature: Optional[float] = None,
        think_level: Optional[ThinkLevel] = None,
        stream: bool = False,
        max_tokens: int = 2000,
    ) -> Tuple[Dict, str]:
        """Build Ollama API payload. Single source of truth for all methods.

        Args:
            messages: Chat messages
            model: Optional model override
            seed: Optional random seed
            temperature: Optional temperature override
            think_level: Abstract thinking level (mapped to model-specific value)
            stream: Whether to stream response
            max_tokens: Maximum tokens to generate

        Returns:
            (payload, active_model) tuple
        """
        active_model = model or self.model

        options = {
            "temperature": temperature if temperature is not None else self.temperature,
            "num_predict": max_tokens,
        }
        if seed is not None:
            options["seed"] = seed

        payload = {
            "model": active_model,
            "messages": messages,
            "stream": stream,
            "options": options,
        }

        # Map abstract ThinkLevel to model-specific value
        think_value = self._map_think_level(think_level, active_model)
        if think_value is not None:
            payload["think"] = think_value

        return payload, active_model

    def _map_think_level(
        self, level: Optional[ThinkLevel], model: str
    ) -> Optional[Union[str, bool]]:
        """Map abstract ThinkLevel to model-specific value.

        - OFF/None: No think parameter (model default, usually no thinking)
        - LOW/MEDIUM/HIGH: Model-specific thinking values
        - Models that don't support thinking: always None
        """
        if level is None or level == ThinkLevel.OFF:
            return None  # No think parameter - let model use its default

        # Models that don't support Ollama's think parameter
        model_lower = model.lower()
        if "qwen" in model_lower:
            return None

        return level in (ThinkLevel.MEDIUM, ThinkLevel.HIGH)

    def complete(self, messages: List[Dict[str, str]], max_tokens: int = 300, model: Optional[str] = None, seed: Optional[int] = None, temperature: Optional[float] = None, think_level: Optional[ThinkLevel] = None) -> ProviderResponse:
        """Generate completion using Ollama API. Returns immutable ProviderResponse."""
        payload, _ = self._build_payload(messages, model, seed, temperature, think_level, stream=False, max_tokens=max_tokens)

        try:
            response = requests.post(
                f"{self.base_url}/api/chat",
                json=payload,
                timeout=self.timeout,
            )

            if response.status_code != 200:
                error = response.text
                return ProviderResponse(
                    text=f"The oracle is silent. (Ollama error {response.status_code}: {error})"
                )

            result = response.json()
            done_reason = result.get("done_reason", "")
            eval_count = result.get("eval_count", 0)
            message = result.get("message", {})
            content = message.get("content", "")
            thinking = message.get("thinking", "")

            # Show thinking in DEBUG mode (styled)
            if thinking and os.getenv("DEBUG"):
                from compass.cli.ui import show_thought
                preview = thinking[:800] + ("..." if len(thinking) > 800 else "")
                show_thought(preview)

            # Log metadata in DEBUG mode
            if os.getenv("DEBUG"):
                truncated = " [TRUNCATED]" if done_reason == "length" else ""
                print(f"[META] tokens={eval_count} done={done_reason}{truncated}")

            return ProviderResponse(
                text=content.strip(),
                thinking=thinking,
                done_reason=done_reason,
                eval_count=eval_count,
            )

        except requests.exceptions.ConnectionError:
            return ProviderResponse(
                text=f"The oracle is silent. (Cannot connect to Ollama at {self.base_url})"
            )
        except requests.exceptions.Timeout:
            return ProviderResponse(
                text=f"The oracle is silent. (Ollama request timed out after {self.timeout}s)"
            )
        except Exception as e:
            return ProviderResponse(text=f"The oracle is silent. ({e})")

    def stream(self, messages: List[Dict[str, str]], max_tokens: int = 300, model: Optional[str] = None, seed: Optional[int] = None, temperature: Optional[float] = None, think_level: Optional[ThinkLevel] = None) -> Generator[str, None, None]:
        """Stream completion token by token from Ollama."""
        self._last_thinking = ""
        self._last_content = ""
        self._last_done_reason = ""
        self._last_eval_count = 0

        payload, _ = self._build_payload(messages, model, seed, temperature, think_level, stream=True, max_tokens=max_tokens)

        try:
            response = requests.post(
                f"{self.base_url}/api/chat",
                json=payload,
                timeout=self.timeout,
                stream=True
            )

            if response.status_code != 200:
                yield f"The oracle is silent. (Ollama error {response.status_code})"
                return

            import json
            import time
            accumulated_content = ""
            done_reason = ""
            eval_count = 0

            for line in response.iter_lines():
                if line:
                    try:
                        chunk = json.loads(line)
                        message = chunk.get("message", {})
                        content = message.get("content", "")
                        if content:
                            accumulated_content += content
                            yield content
                        if chunk.get("done"):
                            done_reason = chunk.get("done_reason", "")
                            eval_count = chunk.get("eval_count", 0)
                            break
                    except json.JSONDecodeError:
                        continue
            self._last_content = accumulated_content
            self._last_done_reason = done_reason or self._last_done_reason
            self._last_eval_count = eval_count

        except requests.exceptions.ConnectionError:
            yield f"The oracle is silent. (Cannot connect to Ollama)"
        except requests.exceptions.Timeout:
            yield f"The oracle is silent. (Timeout)"
        except Exception as e:
            yield f"The oracle is silent. ({e})"

    def complete_with_thinking(
        self,
        messages: List[Dict[str, str]],
        on_thinking: Optional[callable] = None,
        model: Optional[str] = None,
        seed: Optional[int] = None,
        temperature: Optional[float] = None,
        think_level: Optional[ThinkLevel] = None,
        max_tokens: int = 2000,
    ) -> ProviderResponse:
        """Stream completion with real-time thinking display.

        Args:
            messages: Chat messages
            on_thinking: Callback for each thinking chunk
            model: Optional model override
            seed: Optional random seed
            temperature: Optional temperature override
            think_level: Abstract thinking level
            max_tokens: Maximum tokens to generate

        Returns:
            ProviderResponse with text, thinking, done_reason (thinking streamed via callback)
        """
        payload, _ = self._build_payload(messages, model, seed, temperature, think_level, stream=True, max_tokens=max_tokens)

        accumulated_content = ""
        accumulated_thinking = ""
        done_reason = ""
        eval_count = 0

        try:
            response = requests.post(
                f"{self.base_url}/api/chat",
                json=payload,
                timeout=self.timeout,
                stream=True,
            )

            if response.status_code != 200:
                try:
                    error_body = response.text[:500]
                except Exception:
                    error_body = "(no body)"
                return ProviderResponse(
                    text=f"The oracle is silent. (Ollama error {response.status_code}: {error_body})"
                )

            import json

            for line in response.iter_lines():
                if not line:
                    continue

                try:
                    chunk = json.loads(line)
                    message = chunk.get("message", {})

                    # Stream thinking in real-time
                    thinking_chunk = message.get("thinking", "")
                    if thinking_chunk:
                        accumulated_thinking += thinking_chunk
                        if on_thinking:
                            on_thinking(thinking_chunk)

                    # Accumulate content
                    content_chunk = message.get("content", "")
                    if content_chunk:
                        accumulated_content += content_chunk

                    if chunk.get("done"):
                        done_reason = chunk.get("done_reason", "") or done_reason
                        eval_count = chunk.get("eval_count", 0)
                        break

                except json.JSONDecodeError:
                    continue

            # Log metadata in DEBUG mode
            if os.getenv("DEBUG"):
                truncated = " [TRUNCATED]" if done_reason == "length" else ""
                print(f"[META] tokens={eval_count} done={done_reason}{truncated}")

            return ProviderResponse(
                text=accumulated_content.strip(),
                thinking=accumulated_thinking,
                done_reason=done_reason,
                eval_count=eval_count,
            )

        except requests.exceptions.ConnectionError:
            return ProviderResponse(text=f"The oracle is silent. (Cannot connect to Ollama)")
        except requests.exceptions.Timeout:
            return ProviderResponse(text=f"The oracle is silent. (Timeout after {self.timeout}s)")
        except Exception as e:
            return ProviderResponse(text=f"The oracle is silent. ({e})")

    @property
    def supports_streaming(self) -> bool:
        return True

    def health_check(self) -> bool:
        """Check if Ollama is running and the model is available."""
        try:
            response = requests.get(
                f"{self.base_url}/api/tags",
                timeout=5,
            )
            if response.status_code != 200:
                return False

            data = response.json()
            models = [m["name"] for m in data.get("models", [])]
            return self.model in models

        except Exception:
            return False

    def list_models(self) -> List[str]:
        """List available models in Ollama."""
        try:
            response = requests.get(
                f"{self.base_url}/api/tags",
                timeout=5,
            )
            if response.status_code != 200:
                return []

            data = response.json()
            return [m["name"] for m in data.get("models", [])]

        except Exception:
            return []


class LlamaCppProvider(Provider):
    """
    llama.cpp server provider.

    Connects to a llama-server instance via its OpenAI-compatible API.
    The server loads one model at a time, so no model selection is needed
    on the client side -- we query /v1/models to discover what's loaded.

    Configuration:
        LLAMACPP_HOST - Server address (default: http://localhost:8080)

    Example:
        export LLAMACPP_HOST=http://10.106.1.91:8080
    """

    def __init__(
        self,
        base_url: Optional[str] = None,
        temperature: float = 0.7,
        timeout: Optional[int] = None,
        server_name: Optional[str] = None,
    ):
        self.base_url = (base_url or os.getenv("LLAMACPP_HOST", "http://localhost:8080")).rstrip("/")
        self.temperature = temperature
        self.timeout = timeout
        self.server_name = server_name
        self._model_id: Optional[str] = None

    @property
    def model(self) -> str:
        """Discover the loaded model from the server, cache the result."""
        if self._model_id is None:
            try:
                resp = requests.get(f"{self.base_url}/v1/models", timeout=5)
                if resp.status_code == 200:
                    data = resp.json().get("data", [])
                    if data:
                        self._model_id = data[0].get("id", "unknown")
                        return self._model_id
            except Exception:
                pass
            self._model_id = "unknown"
        return self._model_id

    @property
    def name(self) -> str:
        if self.server_name:
            return f"llamacpp/{self.model}@{self.server_name}"
        return f"llamacpp/{self.model}"

    @property
    def supports_streaming(self) -> bool:
        return True

    @staticmethod
    def _convert_messages(messages: List[Dict]) -> List[Dict]:
        """Convert Ollama-style image messages to OpenAI multimodal format.

        Ollama: {"role": "user", "content": "prompt", "images": ["base64..."]}
        OpenAI: {"role": "user", "content": [{"type": "text", ...}, {"type": "image_url", ...}]}
        """
        converted = []
        for msg in messages:
            if "images" in msg:
                content_parts = [{"type": "text", "text": msg.get("content", "")}]
                for img_b64 in msg["images"]:
                    content_parts.append({
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{img_b64}"},
                    })
                converted.append({"role": msg["role"], "content": content_parts})
            else:
                converted.append(msg)
        return converted

    def _build_payload(
        self,
        messages: List[Dict[str, str]],
        max_tokens: int = 300,
        seed: Optional[int] = None,
        temperature: Optional[float] = None,
        stream: bool = False,
    ) -> Dict:
        payload = {
            "messages": self._convert_messages(messages),
            "max_tokens": max_tokens,
            "temperature": temperature if temperature is not None else self.temperature,
            "stream": stream,
        }
        if seed is not None:
            payload["seed"] = seed
        return payload

    def complete(self, messages: List[Dict[str, str]], max_tokens: int = 300, model: Optional[str] = None, seed: Optional[int] = None, temperature: Optional[float] = None, think_level: Optional[ThinkLevel] = None) -> ProviderResponse:
        payload = self._build_payload(messages, max_tokens, seed, temperature, stream=False)

        try:
            response = requests.post(
                f"{self.base_url}/v1/chat/completions",
                json=payload,
                timeout=self.timeout,
            )

            if response.status_code != 200:
                error = response.text[:500]
                return ProviderResponse(
                    text=f"The oracle is silent. (llama.cpp error {response.status_code}: {error})"
                )

            result = response.json()
            choice = result.get("choices", [{}])[0]
            content = choice.get("message", {}).get("content", "")
            finish_reason = choice.get("finish_reason", "")
            usage = result.get("usage", {})
            eval_count = usage.get("completion_tokens", 0)

            if os.getenv("DEBUG"):
                truncated = " [TRUNCATED]" if finish_reason == "length" else ""
                print(f"[META] tokens={eval_count} done={finish_reason}{truncated}")

            return ProviderResponse(
                text=content.strip(),
                done_reason=finish_reason,
                eval_count=eval_count,
            )

        except requests.exceptions.ConnectionError:
            return ProviderResponse(
                text=f"The oracle is silent. (Cannot connect to llama.cpp at {self.base_url})"
            )
        except requests.exceptions.Timeout:
            return ProviderResponse(
                text=f"The oracle is silent. (llama.cpp request timed out after {self.timeout}s)"
            )
        except Exception as e:
            return ProviderResponse(text=f"The oracle is silent. ({e})")

    def stream(self, messages: List[Dict[str, str]], max_tokens: int = 300, model: Optional[str] = None, seed: Optional[int] = None, temperature: Optional[float] = None) -> Generator[str, None, None]:
        import json as _json

        payload = self._build_payload(messages, max_tokens, seed, temperature, stream=True)

        try:
            response = requests.post(
                f"{self.base_url}/v1/chat/completions",
                json=payload,
                timeout=self.timeout,
                stream=True,
            )

            if response.status_code != 200:
                yield f"The oracle is silent. (llama.cpp error {response.status_code})"
                return

            for line in response.iter_lines():
                if not line:
                    continue
                line = line.decode("utf-8") if isinstance(line, bytes) else line
                if not line.startswith("data: "):
                    continue
                data = line[6:]
                if data.strip() == "[DONE]":
                    break
                try:
                    chunk = _json.loads(data)
                    delta = chunk.get("choices", [{}])[0].get("delta", {})
                    content = delta.get("content", "")
                    if content:
                        yield content
                except _json.JSONDecodeError:
                    continue

        except requests.exceptions.ConnectionError:
            yield f"The oracle is silent. (Cannot connect to llama.cpp)"
        except requests.exceptions.Timeout:
            yield f"The oracle is silent. (Timeout)"
        except Exception as e:
            yield f"The oracle is silent. ({e})"

    def health_check(self) -> bool:
        try:
            response = requests.get(f"{self.base_url}/health", timeout=5)
            return response.status_code == 200
        except Exception:
            return False

    def list_models(self) -> List[str]:
        try:
            response = requests.get(f"{self.base_url}/v1/models", timeout=5)
            if response.status_code != 200:
                return []
            data = response.json()
            return [m["id"] for m in data.get("data", [])]
        except Exception:
            return []


def parse_servers(servers_str: str) -> Dict[str, str]:
    """
    Parse OLLAMA_SERVERS format: "name=url,name=url,..."

    Example: "local=http://10.106.1.91:11434,big=http://10.106.1.184:11434"
    Returns: {"local": "http://...", "big": "http://..."}
    """
    if not servers_str:
        return {}

    servers = {}
    for part in servers_str.split(","):
        part = part.strip()
        if "=" in part:
            name, url = part.split("=", 1)
            servers[name.strip()] = url.strip()
    return servers


def parse_ladder(ladder_str: str, servers: Dict[str, str]) -> List[Provider]:
    """
    Parse ladder spec format: "model@server,model@server,...,anthropic:model"

    Example: "qwen3-coder:latest@local,qwen3-coder-next:latest@big,anthropic:sonnet,anthropic:opus"
    Returns: [OllamaProvider(...), OllamaProvider(...), AnthropicProvider(sonnet), AnthropicProvider(opus)]

    Anthropic syntax:
        - "anthropic" → default (sonnet)
        - "anthropic:sonnet" → sonnet
        - "anthropic:opus" → opus

    LlamaCpp syntax:
        - "llamacpp" → default (localhost:8080)
        - "llamacpp@server" → specific server from LLAMACPP_SERVERS

    Deduplicates by provider name (keeps first occurrence).
    """
    if not ladder_str:
        return []

    llamacpp_servers = parse_servers(os.getenv("LLAMACPP_SERVERS", ""))

    providers = []
    seen_names = set()  # Track provider names to deduplicate
    for part in ladder_str.split(","):
        part = part.strip()
        if not part:
            continue

        provider = None
        if part.lower().startswith("anthropic"):
            # Anthropic provider: "anthropic" or "anthropic:model"
            if os.getenv("ANTHROPIC_API_KEY"):
                try:
                    if ":" in part:
                        _, model_name = part.split(":", 1)
                        provider = AnthropicProvider(model=model_name.strip())
                    else:
                        provider = AnthropicProvider()
                except Exception:
                    pass
        elif part.lower().startswith("llamacpp"):
            # LlamaCpp provider: "llamacpp" or "llamacpp@server"
            try:
                if "@" in part:
                    _, server_name = part.split("@", 1)
                    server_name = server_name.strip()
                    server_url = llamacpp_servers.get(server_name)
                    if server_url:
                        provider = LlamaCppProvider(base_url=server_url, server_name=server_name)
                else:
                    default_url = os.getenv("LLAMACPP_HOST") or (list(llamacpp_servers.values())[0] if llamacpp_servers else None)
                    if default_url:
                        provider = LlamaCppProvider(base_url=default_url)
                    else:
                        provider = LlamaCppProvider()
            except Exception:
                pass
        elif "@" in part:
            # Ollama model@server format
            model, server_name = part.rsplit("@", 1)
            server_name = server_name.strip()
            server_url = servers.get(server_name)
            if server_url:
                try:
                    provider = OllamaProvider(model=model.strip(), base_url=server_url, server_name=server_name)
                except Exception:
                    pass
        else:
            # Bare model name - use default server (first in list or OLLAMA_HOST)
            default_url = os.getenv("OLLAMA_HOST") or (list(servers.values())[0] if servers else None)
            if default_url:
                try:
                    provider = OllamaProvider(model=part.strip(), base_url=default_url)
                except Exception:
                    pass

        # Deduplicate by provider name (keep first occurrence)
        if provider and provider.name not in seen_names:
            seen_names.add(provider.name)
            providers.append(provider)

    return providers


def get_provider_by_id(provider_id: str) -> Provider:
    """
    Get a provider by its ID (model@server format).

    Examples:
        get_provider_by_id("qwen3-coder:latest@local")
        get_provider_by_id("qwen3-coder-next:latest@big")
        get_provider_by_id("anthropic")
        get_provider_by_id("anthropic:opus")
        get_provider_by_id("llamacpp")
        get_provider_by_id("llamacpp@local")

    Uses OLLAMA_SERVERS / LLAMACPP_SERVERS env vars for server resolution.
    """
    provider_id = provider_id.strip()

    if provider_id.lower().startswith("anthropic"):
        if ":" in provider_id:
            _, model_name = provider_id.split(":", 1)
            return AnthropicProvider(model=model_name.strip())
        return AnthropicProvider()

    if provider_id.lower().startswith("llamacpp"):
        llamacpp_servers = parse_servers(os.getenv("LLAMACPP_SERVERS", ""))
        if "@" in provider_id:
            _, server_name = provider_id.split("@", 1)
            server_name = server_name.strip()
            server_url = llamacpp_servers.get(server_name)
            if not server_url:
                raise ValueError(f"Unknown llamacpp server '{server_name}'. Available: {list(llamacpp_servers.keys())}")
            return LlamaCppProvider(base_url=server_url, server_name=server_name)
        default_url = os.getenv("LLAMACPP_HOST") or (list(llamacpp_servers.values())[0] if llamacpp_servers else None)
        if default_url:
            return LlamaCppProvider(base_url=default_url)
        return LlamaCppProvider()

    servers = parse_servers(os.getenv("OLLAMA_SERVERS", ""))

    if "@" in provider_id:
        model, server_name = provider_id.rsplit("@", 1)
        server_name = server_name.strip()
        server_url = servers.get(server_name)
        if not server_url:
            raise ValueError(f"Unknown server '{server_name}'. Available: {list(servers.keys())}")
        return OllamaProvider(model=model.strip(), base_url=server_url, server_name=server_name)
    else:
        # Bare model - use default server
        default_url = os.getenv("OLLAMA_HOST") or (list(servers.values())[0] if servers else None)
        if not default_url:
            raise ValueError("No server available. Set OLLAMA_SERVERS or OLLAMA_HOST")
        return OllamaProvider(model=provider_id, base_url=default_url)


def get_provider(provider_type: str = None) -> Provider:
    """
    Get the appropriate provider based on configuration.

    Args:
        provider_type: "anthropic", "ollama", "llamacpp", or None (auto-detect)

    Auto-detection priority:
        1. If LLAMACPP_HOST is set -> llama.cpp
        2. If OLLAMA_HOST or OLLAMA_MODEL is set -> Ollama
        3. If ANTHROPIC_API_KEY is set -> Anthropic
        4. Error if neither configured

    Returns:
        Configured Provider instance
    """
    if provider_type is None:
        # Auto-detect based on environment
        if os.getenv("LLAMACPP_HOST"):
            provider_type = "llamacpp"
        elif os.getenv("OLLAMA_HOST") or os.getenv("OLLAMA_MODEL"):
            provider_type = "ollama"
        elif os.getenv("ANTHROPIC_API_KEY"):
            provider_type = "anthropic"
        else:
            raise ValueError(
                "No LLM provider configured. Set LLAMACPP_HOST for llama.cpp, "
                "ANTHROPIC_API_KEY for Claude, "
                "or OLLAMA_HOST/OLLAMA_MODEL for local Ollama."
            )

    if provider_type == "llamacpp":
        return LlamaCppProvider()
    elif provider_type == "ollama":
        return OllamaProvider()
    elif provider_type == "anthropic":
        return AnthropicProvider()
    else:
        raise ValueError(f"Unknown provider type: {provider_type}")
