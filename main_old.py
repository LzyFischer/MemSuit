"""
SimpleMem — three-stage memory pipeline for LLM agents.

  Stage 1: Semantic Structured Compression  (MemoryBuilder)
  Stage 2: Online Semantic Synthesis        (MemoryBuilder, intra-session)
  Stage 3: Intent-Aware Retrieval Planning  (HybridRetriever + AnswerGenerator)
"""
from typing import List, Optional

import config
from core.answer_generator import AnswerGenerator
from core.hybrid_retriever import HybridRetriever
from core.memory_builder import MemoryBuilder
from database.vector_store import VectorStore
from models.memory_entry import Dialogue, MemoryEntry
from utils.embedding import EmbeddingModel
from utils.llm_client import LLMClient


class SimpleMemSystem:
    def __init__(
        self,
        # LLM overrides
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        base_url: Optional[str] = None,
        enable_thinking: Optional[bool] = None,
        use_streaming: Optional[bool] = None,
        # DB
        db_path: Optional[str] = None,
        table_name: Optional[str] = None,
        clear_db: bool = False,
        # Memory builder
        enable_parallel_processing: Optional[bool] = None,
        max_parallel_workers: Optional[int] = None,
        # Retriever
        enable_planning: Optional[bool] = None,
        enable_reflection: Optional[bool] = None,
        max_reflection_rounds: Optional[int] = None,
        enable_parallel_retrieval: Optional[bool] = None,
        max_retrieval_workers: Optional[int] = None,
    ):
        print("=" * 60)
        print("Initializing SimpleMem")
        print("=" * 60)

        self.llm_client = LLMClient(
            api_key=api_key,
            model=model,
            base_url=base_url,
            enable_thinking=enable_thinking,
            use_streaming=use_streaming,
        )
        self.embedding_model = EmbeddingModel()
        self.vector_store = VectorStore(
            db_path=db_path,
            embedding_model=self.embedding_model,
            table_name=table_name,
        )

        if clear_db:
            print("Clearing existing database...")
            self.vector_store.clear()

        self.memory_builder = MemoryBuilder(
            llm_client=self.llm_client,
            vector_store=self.vector_store,
            enable_parallel_processing=enable_parallel_processing,
            max_parallel_workers=max_parallel_workers,
        )
        self.hybrid_retriever = HybridRetriever(
            llm_client=self.llm_client,
            vector_store=self.vector_store,
            enable_planning=enable_planning,
            enable_reflection=enable_reflection,
            max_reflection_rounds=max_reflection_rounds,
            enable_parallel_retrieval=enable_parallel_retrieval,
            max_retrieval_workers=max_retrieval_workers,
        )
        self.answer_generator = AnswerGenerator(llm_client=self.llm_client)

        print("Ready.\n" + "=" * 60)

    # ------------------------------------------------------------------
    # Write interface
    # ------------------------------------------------------------------

    def add_dialogue(
        self, speaker: str, content: str, timestamp: Optional[str] = None
    ) -> None:
        dialogue_id = (
            self.memory_builder.processed_count
            + len(self.memory_builder.dialogue_buffer)
            + 1
        )
        self.memory_builder.add_dialogue(
            Dialogue(
                dialogue_id=dialogue_id,
                speaker=speaker,
                content=content,
                timestamp=timestamp,
            )
        )

    def add_dialogues(self, dialogues: List[Dialogue]) -> None:
        self.memory_builder.add_dialogues(dialogues)

    def finalize(self) -> None:
        """Flush any remaining buffered dialogues."""
        self.memory_builder.process_remaining()

    # ------------------------------------------------------------------
    # Query interface
    # ------------------------------------------------------------------

    def ask(self, question: str) -> str:
        print("\n" + "=" * 60)
        print(f"Question: {question}")
        print("=" * 60)
        contexts = self.hybrid_retriever.retrieve(question)
        answer = self.answer_generator.generate_answer(question, contexts)
        print(f"\nAnswer: {answer}")
        print("=" * 60 + "\n")
        return answer

    # ------------------------------------------------------------------
    # Debug helpers
    # ------------------------------------------------------------------

    def get_all_memories(self) -> List[MemoryEntry]:
        return self.vector_store.get_all_entries()

    def print_memories(self) -> None:
        memories = self.get_all_memories()
        print(f"\n{'=' * 60}\nAll Memories ({len(memories)} entries)\n{'=' * 60}")
        for i, m in enumerate(memories, 1):
            print(f"\n[{i}] {m.lossless_restatement}")
        print("=" * 60)
