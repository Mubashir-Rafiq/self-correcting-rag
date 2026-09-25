"""Unit tests for Stage 10: RAGSystem composition root, observability, and chat interface."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from langchain_core.messages import AIMessage
from langchain_core.tools import tool
from pydantic import SecretStr

from self_rag.cli import build_parser, main
from self_rag.config import Settings
from self_rag.observability import ExecutionLogger, ObservabilityManager
from self_rag.system import RAGSystem


class TestObservability:
    """Tests for ObservabilityManager and ExecutionLogger."""

    def test_observability_disabled_by_default(self) -> None:
        settings = Settings()
        obs = ObservabilityManager(settings)
        assert obs.enabled is False
        assert obs.get_callbacks() == []
        obs.flush()  # safe no-op

    def test_observability_disabled_when_keys_missing(self) -> None:
        settings = Settings(langfuse_enabled=True, langfuse_public_key=None)
        obs = ObservabilityManager(settings)
        assert obs.enabled is False
        assert obs.get_callbacks() == []

    def test_observability_initializes_with_valid_credentials(self) -> None:
        settings = Settings(
            langfuse_enabled=True,
            langfuse_public_key=SecretStr("pk-lf-test"),
            langfuse_secret_key=SecretStr("sk-lf-test"),
        )
        mock_client = MagicMock()
        mock_client.auth_check.return_value = True
        mock_handler = MagicMock()

        with (
            patch("langfuse.Langfuse", return_value=mock_client),
            patch("langfuse.langchain.CallbackHandler", return_value=mock_handler),
        ):
            obs = ObservabilityManager(settings)
            assert obs.enabled is True
            assert obs.get_callbacks() == [mock_handler]
            obs.flush()
            assert mock_client.flush.called

    def test_execution_logger_disabled_passthrough(self) -> None:
        logger = ExecutionLogger(enabled=False)
        fn = MagicMock(return_value="result")
        wrapped = logger.wrap_node("test_node", fn)
        assert wrapped("input_val") == "result"
        assert fn.called

    def test_execution_logger_enabled_logs_events(self) -> None:
        logger = ExecutionLogger(enabled=True, max_chars=50)
        fn = MagicMock(return_value="long result " * 20)
        wrapped = logger.wrap_node("test_node", fn)
        with patch("self_rag.observability.logger.info") as mock_info:
            res = wrapped("input_val")
            assert res == "long result " * 20
            assert mock_info.call_count == 2
            # Verify start log
            assert mock_info.call_args_list[0][0][1] == "test_node"
            # Verify end log
            assert mock_info.call_args_list[1][0][1] == "test_node"


class TestRAGSystem:
    """Tests for the RAGSystem composition root."""

    @pytest.fixture()
    def mock_components(self) -> dict[str, Any]:
        from self_rag.agent.graph import create_sqlite_checkpointer

        @tool
        def dummy_search(query: str) -> str:
            """Dummy search tool."""
            return "[]"

        mock_vm = MagicMock()
        mock_vm.collection_exists.return_value = True
        mock_vm.count.return_value = 5

        mock_ps = MagicMock()
        mock_llm = MagicMock()
        checkpointer = create_sqlite_checkpointer(":memory:")
        mock_obs = MagicMock()
        mock_obs.get_callbacks.return_value = []

        return {
            "vm": mock_vm,
            "ps": mock_ps,
            "llm": mock_llm,
            "tool": dummy_search,
            "checkpointer": checkpointer,
            "obs": mock_obs,
        }

    def test_create_thread_id_and_config(self, mock_components: dict[str, Any]) -> None:
        settings = Settings()
        system = RAGSystem(
            settings=settings,
            vector_store_manager=mock_components["vm"],
            parent_store=mock_components["ps"],
            llm=mock_components["llm"],
            tools=[mock_components["tool"]],
            checkpointer=mock_components["checkpointer"],
            observability=mock_components["obs"],
        )

        tid1 = system.create_thread_id()
        tid2 = system.create_thread_id()
        assert tid1 != tid2
        assert len(tid1) == 32

        config = system.get_config(tid1)
        assert config["configurable"]["thread_id"] == tid1
        assert config["recursion_limit"] == settings.graph_recursion_limit

        system.close()
        assert mock_components["obs"].flush.called

    def test_context_manager_lifecycle(self, mock_components: dict[str, Any]) -> None:
        settings = Settings()
        with RAGSystem(
            settings=settings,
            vector_store_manager=mock_components["vm"],
            parent_store=mock_components["ps"],
            llm=mock_components["llm"],
            tools=[mock_components["tool"]],
            checkpointer=mock_components["checkpointer"],
            observability=mock_components["obs"],
        ) as system:
            assert system.settings == settings

        assert mock_components["obs"].flush.called

    def test_chat_new_question_flow(self, mock_components: dict[str, Any]) -> None:
        settings = Settings()
        system = RAGSystem(
            settings=settings,
            vector_store_manager=mock_components["vm"],
            parent_store=mock_components["ps"],
            llm=mock_components["llm"],
            tools=[mock_components["tool"]],
            checkpointer=mock_components["checkpointer"],
            observability=mock_components["obs"],
        )

        mock_main_graph = MagicMock()
        system.main_graph = mock_main_graph

        # Mock initial state (no interrupt)
        initial_state = MagicMock()
        initial_state.next = ()
        post_state = MagicMock()
        post_state.next = ()
        post_state.values = {"messages": [AIMessage(content="Grounded assistant answer.")]}

        mock_main_graph.get_state.side_effect = [initial_state, post_state]

        res = system.chat("What is RAG?", thread_id="t1")

        assert res["answer"] == "Grounded assistant answer."
        assert res["is_clarification"] is False
        assert res["thread_id"] == "t1"
        assert mock_main_graph.invoke.called

    def test_chat_clarification_interrupt_and_continuation_flow(
        self, mock_components: dict[str, Any]
    ) -> None:
        settings = Settings()
        system = RAGSystem(
            settings=settings,
            vector_store_manager=mock_components["vm"],
            parent_store=mock_components["ps"],
            llm=mock_components["llm"],
            tools=[mock_components["tool"]],
            checkpointer=mock_components["checkpointer"],
            observability=mock_components["obs"],
        )

        mock_main_graph = MagicMock()
        system.main_graph = mock_main_graph

        # 1. State is currently paused at request_clarification
        paused_state = MagicMock()
        paused_state.next = ("request_clarification",)

        # After resume, finishes execution
        resumed_state = MagicMock()
        resumed_state.next = ()
        resumed_state.values = {
            "messages": [AIMessage(content="Resolved answer after clarification.")]
        }

        mock_main_graph.get_state.side_effect = [paused_state, resumed_state]

        res = system.chat("Specific clarification reply", thread_id="t1")

        assert mock_main_graph.update_state.called
        update_args = mock_main_graph.update_state.call_args[0]
        assert update_args[1]["messages"][0].content == "Specific clarification reply"
        assert res["answer"] == "Resolved answer after clarification."
        assert res["is_clarification"] is False


class TestCliChatCommand:
    """Tests for self-rag chat CLI command."""

    def test_parser_registers_chat(self) -> None:
        parser = build_parser()
        args = parser.parse_args(["chat", "--thread-id", "my-thread"])
        assert args.command == "chat"
        assert args.thread_id == "my-thread"

    def test_chat_cli_runs_and_exits(self) -> None:
        mock_system = MagicMock()
        mock_system.__enter__.return_value = mock_system
        mock_system.create_thread_id.return_value = "tid_123"
        mock_system.chat.return_value = {
            "answer": "Hello from RAG!",
            "is_clarification": False,
        }

        with (
            patch("self_rag.system.RAGSystem", return_value=mock_system),
            patch("builtins.input", side_effect=["Hi", "/exit"]),
            patch("builtins.print") as mock_print,
        ):
            exit_code = main(["chat"])
            assert exit_code == 0
            assert mock_system.chat.called
            assert any("Hello from RAG!" in str(call) for call in mock_print.call_args_list)
