"""Tests for Qdrant server mode and docker-compose deployment configuration."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import yaml  # type: ignore[import-untyped]
from pydantic import SecretStr

from self_rag.config import Settings
from self_rag.storage.vector_store import VectorStoreManager


class TestQdrantServerConfiguration:
    def test_docker_compose_validity(self) -> None:
        compose_path = Path("docker-compose.yml")
        assert compose_path.exists(), "docker-compose.yml must exist at project root"

        content = yaml.safe_load(compose_path.read_text(encoding="utf-8"))
        assert "services" in content
        assert "qdrant" in content["services"]
        qdrant_service = content["services"]["qdrant"]
        assert "image" in qdrant_service
        assert "qdrant/qdrant" in qdrant_service["image"]
        assert "ports" in qdrant_service
        # Must expose port 6333
        assert any("6333:6333" in p for p in qdrant_service["ports"])

    def test_settings_qdrant_url(self) -> None:
        settings = Settings(
            qdrant_url="http://qdrant.internal:6333",
            qdrant_api_key=SecretStr("secret-token"),
        )
        assert settings.qdrant_url == "http://qdrant.internal:6333"
        assert settings.qdrant_api_key is not None
        assert settings.qdrant_api_key.get_secret_value() == "secret-token"

    @patch("self_rag.storage.vector_store.QdrantClient")
    def test_vector_store_connects_to_server_url(self, mock_qdrant_client_cls: MagicMock) -> None:
        settings = Settings(
            qdrant_url="http://localhost:6333",
            qdrant_api_key=SecretStr("token-123"),
        )
        vm = VectorStoreManager(settings=settings)
        # Access client
        _ = vm.client

        # Verify QdrantClient was instantiated with url and api_key
        mock_qdrant_client_cls.assert_called_once_with(
            url="http://localhost:6333",
            api_key="token-123",
        )
