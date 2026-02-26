"""Tests for infra.config -- model configuration dataclasses."""

import pytest

from infra.config import (
    AppConfig,
    ChatModelConfig,
    EmbeddingModelConfig,
    ProviderConfig,
    default_config,
)


class TestChatModelConfig:
    def test_defaults(self):
        cfg = ChatModelConfig(
            model_name="test-model",
            context_limit=1000,
            max_output_tokens=500,
        )
        assert cfg.temperature == 0.3
        assert cfg.timeout == 300

    def test_custom_values(self):
        cfg = ChatModelConfig(
            model_name="m",
            context_limit=2000,
            max_output_tokens=1000,
            temperature=0.7,
            timeout=600,
        )
        assert cfg.temperature == 0.7
        assert cfg.timeout == 600

    def test_frozen(self):
        cfg = ChatModelConfig(model_name="m", context_limit=1, max_output_tokens=1)
        with pytest.raises(AttributeError):
            cfg.model_name = "other"


class TestEmbeddingModelConfig:
    def test_defaults(self):
        cfg = EmbeddingModelConfig(model_name="emb")
        assert cfg.timeout == 120

    def test_frozen(self):
        cfg = EmbeddingModelConfig(model_name="emb")
        with pytest.raises(AttributeError):
            cfg.model_name = "other"


class TestProviderConfig:
    def test_default_no_base_url(self):
        cfg = ProviderConfig()
        assert cfg.base_url is None

    def test_with_base_url(self):
        cfg = ProviderConfig(base_url="http://localhost:9999/v1")
        assert cfg.base_url == "http://localhost:9999/v1"


class TestAppConfig:
    def test_default_config_structure(self):
        """Verify default_config has all expected fields with sensible types."""
        cfg = default_config
        # Chat configs have required fields
        assert isinstance(cfg.openai_chat.model_name, str) and cfg.openai_chat.model_name
        assert cfg.openai_chat.context_limit > 0
        assert cfg.openai_chat.max_output_tokens > 0
        assert isinstance(cfg.lmstudio_chat.model_name, str) and cfg.lmstudio_chat.model_name
        assert cfg.lmstudio_chat.context_limit > 0
        assert cfg.lmstudio_chat.max_output_tokens > 0
        # Embedding configs
        assert isinstance(cfg.openai_embedding.model_name, str) and cfg.openai_embedding.model_name
        assert isinstance(cfg.lmstudio_embedding.model_name, str) and cfg.lmstudio_embedding.model_name
        # Provider configs
        assert cfg.lmstudio.base_url is not None
        assert cfg.openai.base_url is None

    def test_default_matches_fresh_appconfig(self):
        """default_config equals a freshly constructed AppConfig()."""
        fresh = AppConfig()
        assert default_config.openai_chat == fresh.openai_chat
        assert default_config.lmstudio_chat == fresh.lmstudio_chat
        assert default_config.openai_embedding == fresh.openai_embedding
        assert default_config.lmstudio_embedding == fresh.lmstudio_embedding
        assert default_config.openai == fresh.openai
        assert default_config.lmstudio == fresh.lmstudio

    def test_override_single_model(self):
        """Can override one model without affecting others."""
        custom_chat = ChatModelConfig(
            model_name="custom-model",
            context_limit=64_000,
            max_output_tokens=16_384,
        )
        cfg = AppConfig(openai_chat=custom_chat)
        assert cfg.openai_chat.model_name == "custom-model"
        assert cfg.openai_chat.context_limit == 64_000
        # Other defaults unaffected
        assert cfg.lmstudio_chat == default_config.lmstudio_chat
        assert cfg.lmstudio == default_config.lmstudio

    def test_frozen(self):
        with pytest.raises(AttributeError):
            default_config.openai_chat = ChatModelConfig(
                model_name="x", context_limit=1, max_output_tokens=1
            )
