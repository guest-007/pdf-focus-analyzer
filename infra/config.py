"""Central configuration for model definitions and provider settings."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class ChatModelConfig:
    """Configuration for a single chat model."""

    model_name: str
    context_limit: int
    max_output_tokens: int
    temperature: float = 0.3
    timeout: int = 300  # seconds, used by HTTP-based providers


@dataclass(frozen=True)
class EmbeddingModelConfig:
    """Configuration for a single embedding model."""

    model_name: str
    timeout: int = 120  # seconds


@dataclass(frozen=True)
class ProviderConfig:
    """Connection details for a provider."""

    base_url: str | None = None  # None means use SDK default (e.g. OpenAI)


@dataclass(frozen=True)
class AppConfig:
    """Top-level application configuration."""

    openai: ProviderConfig = field(default_factory=ProviderConfig)
    lmstudio: ProviderConfig = field(
        default_factory=lambda: ProviderConfig(base_url="http://localhost:1234/v1")
    )

    openai_chat: ChatModelConfig = field(
        default_factory=lambda: ChatModelConfig(
            model_name="gpt-4.1",
            context_limit=128_000,
            max_output_tokens=32_768,
        )
    )
    lmstudio_chat: ChatModelConfig = field(
        default_factory=lambda: ChatModelConfig(
            model_name="qwen/qwen3.5-35b-a3b",
            context_limit=4_096,
            max_output_tokens=4_096,
            timeout=900,
        )
    )
    openai_embedding: EmbeddingModelConfig = field(
        default_factory=lambda: EmbeddingModelConfig(
            model_name="text-embedding-3-small",
        )
    )
    lmstudio_embedding: EmbeddingModelConfig = field(
        default_factory=lambda: EmbeddingModelConfig(
            model_name="text-embedding-nomic-embed-text-v1.5",
        )
    )


# Module-level singleton -- import this from anywhere.
default_config = AppConfig()
