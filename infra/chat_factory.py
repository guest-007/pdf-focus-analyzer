# chat_factory.py

import re
import subprocess
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, List, Literal, Optional, TypedDict, cast

from dotenv import load_dotenv

from infra.config import (
    AppConfig,
    ChatModelConfig,
    EmbeddingModelConfig,
    ProviderConfig,
    default_config,
)

if TYPE_CHECKING:
    from openai.types.chat import ChatCompletionMessageParam

load_dotenv()  # Load .env file


ChatRole = Literal["system", "user", "assistant"]


class ChatMessage(TypedDict):
    role: ChatRole
    content: str


class ChatModel(ABC):
    context_limit: int      # max input + output tokens
    max_output_tokens: int  # max completion tokens the API accepts

    @abstractmethod
    def generate(
        self,
        messages: List[ChatMessage],
        response_format: Optional[dict] = None,
        max_tokens: Optional[int] = None,
    ) -> str:
        pass


class EmbeddingModel(ABC):
    @abstractmethod
    def embed(self, texts: List[str]) -> List[List[float]]:
        pass


class OpenAIChat(ChatModel):
    def __init__(self, config: ChatModelConfig):
        from openai import OpenAI

        self.client = OpenAI()  # auto reads OPENAI_API_KEY
        self.model_name = config.model_name
        self.context_limit = config.context_limit
        self.max_output_tokens = config.max_output_tokens
        self.temperature = config.temperature

    def generate(
        self,
        messages: List[ChatMessage],
        response_format: Optional[dict] = None,
        max_tokens: Optional[int] = None,
    ) -> str:
        kwargs: dict = {
            "model": self.model_name,
            "messages": cast(List["ChatCompletionMessageParam"], messages),
            "temperature": self.temperature,
        }
        if max_tokens is not None:
            kwargs["max_tokens"] = max_tokens
        if response_format is not None:
            kwargs["response_format"] = response_format
        response = self.client.chat.completions.create(**kwargs)
        return response.choices[0].message.content or ""


class OpenAIEmbedding(EmbeddingModel):
    def __init__(self, config: EmbeddingModelConfig):
        from openai import OpenAI

        self.client = OpenAI()
        self.model_name = config.model_name

    def embed(self, texts: List[str]) -> List[List[float]]:
        response = self.client.embeddings.create(
            model=self.model_name,
            input=texts,
        )
        sorted_data = sorted(response.data, key=lambda x: x.index)
        return [item.embedding for item in sorted_data]


class LMStudioChat(ChatModel):
    def __init__(
        self, chat_config: ChatModelConfig, provider_config: ProviderConfig
    ):
        import requests

        self.base_url = provider_config.base_url.rstrip("/")
        self.model_name = chat_config.model_name
        self.context_limit = chat_config.context_limit
        self.max_output_tokens = chat_config.max_output_tokens
        self.temperature = chat_config.temperature
        self.timeout = chat_config.timeout
        self.requests = requests

    def generate(
        self,
        messages: List[ChatMessage],
        response_format: Optional[dict] = None,
        max_tokens: Optional[int] = None,
    ) -> str:
        payload: dict = {
            "model": self.model_name,
            "messages": messages,
            "temperature": self.temperature,
            "chat_template_kwargs": {"enable_thinking": False},
        }
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens
        if response_format is not None:
            payload["response_format"] = response_format
        response = self.requests.post(
            f"{self.base_url}/chat/completions",
            json=payload,
            timeout=self.timeout,
        )
        response.raise_for_status()
        content = response.json()["choices"][0]["message"]["content"]

        # Remove <think>...</think> blocks
        content = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()
        return content


class LMStudioEmbedding(EmbeddingModel):
    def __init__(
        self, config: EmbeddingModelConfig, provider_config: ProviderConfig
    ):
        import requests

        self.base_url = provider_config.base_url.rstrip("/")
        self.model_name = config.model_name
        self.timeout = config.timeout
        self.requests = requests

    def embed(self, texts: List[str]) -> List[List[float]]:
        response = self.requests.post(
            f"{self.base_url}/embeddings",
            json={
                "model": self.model_name,
                "input": texts,
            },
            timeout=self.timeout,
        )
        response.raise_for_status()
        data = response.json()["data"]
        sorted_data = sorted(data, key=lambda x: x["index"])
        return [item["embedding"] for item in sorted_data]


class ChatFactory:
    """Factory with configurable defaults + LM Studio server controls."""

    def __init__(self, config: AppConfig | None = None):
        self.config = config or default_config

    def create(self, provider: str = "lmstudio") -> ChatModel:
        provider = provider.lower().strip()

        if provider == "openai":
            return OpenAIChat(self.config.openai_chat)
        if provider == "lmstudio":
            return LMStudioChat(self.config.lmstudio_chat, self.config.lmstudio)
        raise ValueError(f"Unsupported provider: {provider}")

    def create_embedder(self, provider: str = "openai") -> EmbeddingModel:
        provider = provider.lower().strip()
        if provider == "openai":
            return OpenAIEmbedding(self.config.openai_embedding)
        if provider == "lmstudio":
            return LMStudioEmbedding(
                self.config.lmstudio_embedding, self.config.lmstudio
            )
        raise ValueError(f"Unsupported embedding provider: {provider}")

    def _is_lms_server_running(self, timeout: int = 2) -> bool:
        """Check whether LM Studio server responds to /models."""
        import requests

        base_url = self.config.lmstudio.base_url
        try:
            r = requests.get(f"{base_url}/models", timeout=timeout)
            return r.status_code == 200
        except Exception:
            return False

    def ensure_lms_ready(self) -> None:
        """Check that LM Studio server is running and the right model is loaded.

        Does NOT load models — load them via the LM Studio GUI instead.
        Uses ``lms ps`` to check what's actually in memory (the /v1/models
        API lists all downloaded models, not just loaded ones).
        """
        if not self._is_lms_server_running():
            raise RuntimeError(
                "LM Studio server is not running.\n"
                "Open LM Studio, load a model, and start the server (Developer tab)."
            )

        # lms ps shows models actually loaded in memory
        try:
            result = subprocess.run(
                ["lms", "ps"],
                capture_output=True, text=True, check=False, timeout=15,
            )
            ps_output = result.stdout
        except Exception:
            ps_output = ""

        model_name = self.config.lmstudio_chat.model_name
        base_key = model_name.split("@")[0]

        if base_key in ps_output:
            return  # model is loaded in memory

        # Build a helpful error message
        loaded_lines = [
            line.strip() for line in ps_output.splitlines()
            if line.strip() and not line.startswith(("─", "TYPE", "LLM", "EMBEDDING"))
        ]
        if loaded_lines:
            loaded_str = "\n    ".join(loaded_lines)
            raise RuntimeError(
                f"Wrong model loaded in LM Studio.\n"
                f"  Expected: {model_name}\n"
                f"  Loaded:\n    {loaded_str}\n"
                f"Load the correct model in the LM Studio GUI."
            )
        else:
            raise RuntimeError(
                f"No model loaded in LM Studio.\n"
                f"  Expected: {model_name}\n"
                f"Load it in the LM Studio GUI, then try again."
            )

    @staticmethod
    def stop_lms_server() -> subprocess.CompletedProcess:
        """Stop LM Studio server via lms CLI."""
        return subprocess.run(
            ["lms", "server", "stop"],
            capture_output=True,
            text=True,
            check=False,
        )


# Helper message functions
def system_message(content: str) -> ChatMessage:
    return {"role": "system", "content": content}


def human_message(content: str) -> ChatMessage:
    return {"role": "user", "content": content}


def assistant_message(content: str) -> ChatMessage:
    return {"role": "assistant", "content": content}


def _fix_schema_for_strict(schema: dict) -> dict:
    """Add additionalProperties:false and remove defaults for OpenAI strict mode."""
    if schema.get("type") == "object" and "properties" in schema:
        schema["additionalProperties"] = False
        # All properties must be required for strict mode
        schema["required"] = list(schema["properties"].keys())
        for prop in schema["properties"].values():
            prop.pop("default", None)
            _fix_schema_for_strict(prop)
    # Fix nested $defs
    for defn in schema.get("$defs", {}).values():
        _fix_schema_for_strict(defn)
    # Fix items in arrays
    if "items" in schema:
        _fix_schema_for_strict(schema["items"])
    return schema


def make_response_format(model_class) -> dict:
    """Build an OpenAI-compatible response_format dict from a Pydantic model."""
    schema = _fix_schema_for_strict(model_class.model_json_schema())
    return {
        "type": "json_schema",
        "json_schema": {
            "name": model_class.__name__,
            "strict": True,
            "schema": schema,
        },
    }


# Example usage
if __name__ == "__main__":
    factory = ChatFactory()

    provider = input("Choose model (lmstudio/openai): ").strip().lower() or "lmstudio"
    chat_model = factory.create(provider)

    messages = [
        system_message("You are a helpful assistant."),
        human_message("Hello, how are you?"),
    ]

    response = chat_model.generate(messages)
    print(response)
