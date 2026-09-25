"""Composition root wiring storage, language models, checkpointer, and agent graphs."""

from __future__ import annotations

import contextlib
import uuid
from collections.abc import Generator
from types import TracebackType
from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import BaseTool
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph.state import CompiledStateGraph

from self_rag.agent.graph import (
    build_agent_subgraph,
    build_main_graph,
    create_sqlite_checkpointer,
)
from self_rag.agent.state import AgentState, State
from self_rag.agent.tools import ToolFactory
from self_rag.config import Settings, get_settings
from self_rag.llm.factory import build_llm
from self_rag.observability import ExecutionLogger, ObservabilityManager
from self_rag.storage.embeddings import get_dense_embeddings, get_sparse_embeddings
from self_rag.storage.parent_store import ParentStore
from self_rag.storage.reranker import BaseReranker, FastEmbedReranker
from self_rag.storage.vector_store import VectorStoreManager


class RAGSystem:
    """The central composition root of the Self-Correcting RAG application."""

    def __init__(
        self,
        settings: Settings | None = None,
        *,
        vector_store_manager: VectorStoreManager | None = None,
        parent_store: ParentStore | None = None,
        llm: BaseChatModel | None = None,
        grader_llm: BaseChatModel | None = None,
        reranker: BaseReranker | None = None,
        tools: list[BaseTool] | None = None,
        checkpointer: BaseCheckpointSaver[Any] | None = None,
        observability: ObservabilityManager | None = None,
        execution_logger: ExecutionLogger | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self._owns_vm = vector_store_manager is None
        self.vector_store_manager = vector_store_manager or VectorStoreManager(
            settings=self.settings
        )
        self.parent_store = parent_store or ParentStore(settings=self.settings)

        self.observability = observability or ObservabilityManager(settings=self.settings)
        self.execution_logger = execution_logger or ExecutionLogger(
            enabled=self.settings.execution_logging_enabled,
            max_chars=self.settings.execution_log_max_chars,
        )

        self.checkpointer = checkpointer or create_sqlite_checkpointer(
            self.settings.checkpoint_db_path
        )
        self.llm = llm or build_llm(self.settings)

        self.grader_llm: BaseChatModel | None
        if grader_llm is not None:
            self.grader_llm = grader_llm
        elif self.settings.judge_model:
            judge: BaseChatModel | None = None
            with contextlib.suppress(Exception):
                judge = build_llm(self.settings, model=self.settings.judge_model)
            self.grader_llm = judge
        else:
            self.grader_llm = None

        if reranker is not None:
            self.reranker: BaseReranker | None = reranker
        elif self.settings.reranker_enabled:
            self.reranker = FastEmbedReranker(model_name=self.settings.reranker_model)
        else:
            self.reranker = None

        if tools is None:
            dense = get_dense_embeddings(self.settings)
            sparse = get_sparse_embeddings(self.settings)
            vector_store = self.vector_store_manager.get_vector_store(
                embedding=dense, sparse_embedding=sparse
            )
            tool_factory = ToolFactory(
                collection=vector_store,
                parent_store=self.parent_store,
                default_limit=self.settings.retrieval_k,
                score_threshold=self.settings.hybrid_score_floor,
                reranker=self.reranker,
            )
            self.tools: list[BaseTool] = tool_factory.create_tools()
        else:
            self.tools = tools

        self.subgraph: CompiledStateGraph[AgentState, None, AgentState, AgentState] = (
            build_agent_subgraph(
                llm=self.llm,
                tools=self.tools,
                grader_llm=self.grader_llm,
                self_correction_enabled=self.settings.self_correction_enabled,
                max_correction_retries=self.settings.max_correction_retries,
                max_iterations=self.settings.max_iterations,
                max_tool_calls=self.settings.max_tool_calls,
                base_token_threshold=self.settings.base_token_threshold,
                token_growth_factor=self.settings.token_growth_factor,
            )
        )

        self.main_graph: CompiledStateGraph[State, None, State, State] = build_main_graph(
            llm=self.llm,
            tools=self.tools,
            grader_llm=self.grader_llm,
            self_correction_enabled=self.settings.self_correction_enabled,
            max_correction_retries=self.settings.max_correction_retries,
            checkpointer=self.checkpointer,
            max_subquestions=self.settings.max_subquestions,
            main_history_messages_to_keep=self.settings.main_history_messages_to_keep,
            max_iterations=self.settings.max_iterations,
            max_tool_calls=self.settings.max_tool_calls,
            base_token_threshold=self.settings.base_token_threshold,
            token_growth_factor=self.settings.token_growth_factor,
        )

    def create_thread_id(self) -> str:
        """Create a new unique conversation thread identifier."""
        return uuid.uuid4().hex

    def get_config(self, thread_id: str) -> RunnableConfig:
        """Return the LangGraph runnable configuration for a session thread."""
        cfg: RunnableConfig = {
            "configurable": {"thread_id": thread_id},
            "recursion_limit": self.settings.graph_recursion_limit,
        }
        callbacks = self.observability.get_callbacks()
        if callbacks:
            cfg["callbacks"] = callbacks
        return cfg

    def chat(self, message: str, thread_id: str) -> dict[str, Any]:
        """Send a message in a conversation thread and return the assistant response.

        Handles new user questions as well as clarification continuations when
        paused at an interrupt point.
        """
        config = self.get_config(thread_id)
        current_state = self.main_graph.get_state(config)

        if current_state.next and "request_clarification" in current_state.next:
            self.main_graph.update_state(
                config,
                {"messages": [HumanMessage(content=message)]},
                as_node="request_clarification",
            )
            self.main_graph.invoke(None, config=config)
        else:
            input_state: State = {"messages": [HumanMessage(content=message)]}
            self.main_graph.invoke(input_state, config=config)

        post_state = self.main_graph.get_state(config)
        is_clarification = bool(post_state.next and "request_clarification" in post_state.next)

        messages = post_state.values.get("messages", [])
        last_ai_message = ""
        for m in reversed(messages):
            if isinstance(m, AIMessage) and m.content:
                last_ai_message = str(m.content).strip()
                break

        return {
            "answer": last_ai_message,
            "is_clarification": is_clarification,
            "thread_id": thread_id,
            "state": post_state.values,
        }

    def chat_stream(
        self,
        message: str,
        thread_id: str,
    ) -> Generator[str, None, None]:
        """Stream a response for a message in a conversation thread.

        Yields intermediate status milestones as nodes execute and ends with the
        complete final assistant answer.
        """
        config = self.get_config(thread_id)
        current_state = self.main_graph.get_state(config)

        stream_input: Any
        if current_state.next and "request_clarification" in current_state.next:
            self.main_graph.update_state(
                config,
                {"messages": [HumanMessage(content=message)]},
                as_node="request_clarification",
            )
            stream_input = None
        else:
            stream_input = {"messages": [HumanMessage(content=message)]}

        yield "🔍 Analyzing question..."

        final_answer = ""
        for event in self.main_graph.stream(stream_input, config=config, stream_mode="updates"):
            if "summarize_history" in event:
                yield "🧠 Managing conversation history..."
            elif "rewrite_query" in event:
                rw = event["rewrite_query"]
                if rw.get("questionIsClear") is False:
                    clarification = ""
                    for m in rw.get("messages", []):
                        if getattr(m, "name", None) == "clarification":
                            clarification = str(m.content).strip()
                            break
                    if clarification:
                        final_answer = clarification
                        yield clarification
                        return
                else:
                    qs = rw.get("rewrittenQuestions", [])
                    if qs:
                        yield f"🔎 Searching evidence for: *{qs[0]}*..."
            elif "agent" in event:
                yield "📚 Retrieving and verifying document evidence..."
            elif "aggregate_answers" in event:
                msgs = event["aggregate_answers"].get("messages", [])
                if msgs and msgs[0].content:
                    final_answer = str(msgs[0].content).strip()

        if final_answer:
            yield final_answer
        else:
            post_state = self.main_graph.get_state(config)
            msgs = post_state.values.get("messages", [])
            for m in reversed(msgs):
                if isinstance(m, AIMessage) and m.content:
                    final_answer = str(m.content).strip()
                    break
            yield final_answer or "No answer could be generated."

    def close(self) -> None:
        """Release underlying resources."""
        if self._owns_vm:
            self.vector_store_manager.close()
        self.observability.flush()

    def __enter__(self) -> RAGSystem:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        self.close()
