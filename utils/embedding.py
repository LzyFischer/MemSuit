"""
Embedding model wrapper (SentenceTransformers).
Supports standard models and Qwen3-Embedding variants.
"""
from typing import Any, List, Optional

import numpy as np
import logging

import config


class EmbeddingModel:
    def __init__(self, model_name: Optional[str] = None):
        self.model_name = model_name or config.EMBEDDING_MODEL
        self.model: Any = None
        self.dimension: int = 0
        self.model_type: str = ""
        self.supports_query_prompt: bool = False

        print(f"Loading embedding model: {self.model_name}")
        if self.model_name.lower().startswith("qwen3"):
            self._init_qwen3()
        else:
            self._init_standard()

    def _init_qwen3(self) -> None:
        from sentence_transformers import SentenceTransformer
        model_map = {
            "qwen3-0.6b": "Qwen/Qwen3-Embedding-0.6B",
            "qwen3-4b": "Qwen/Qwen3-Embedding-4B",
            "qwen3-8b": "Qwen/Qwen3-Embedding-8B",
        }
        path = model_map.get(self.model_name, self.model_name)
        logging.set_verbosity_error()
        try:
            self.model = SentenceTransformer(
                path,
                model_kwargs={"attn_implementation": "flash_attention_2", "device_map": "auto"},
                tokenizer_kwargs={"padding_side": "left"},
                trust_remote_code=True,
            )
            print("Qwen3 loaded with flash_attention_2")
        except Exception:
            self.model = SentenceTransformer(path, trust_remote_code=True)
        logging.set_verbosity_warning()
        self.dimension = self.model.get_sentence_embedding_dimension()
        self.model_type = "qwen3_sentence_transformer"
        self.supports_query_prompt = "query" in getattr(self.model, "prompts", {})
        print(f"Qwen3 embedding dim: {self.dimension}")

    def _init_standard(self) -> None:
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(self.model_name, trust_remote_code=True)
        self.dimension = self.model.get_sentence_embedding_dimension()
        self.model_type = "sentence_transformer"
        print(f"Embedding dim: {self.dimension}")

    def encode(self, texts: List[str], is_query: bool = False) -> np.ndarray:
        if isinstance(texts, str):
            texts = [texts]
        if self.model_type == "qwen3_sentence_transformer" and self.supports_query_prompt and is_query:
            return self.model.encode(texts, prompt_name="query", show_progress_bar=False, normalize_embeddings=True)
        return self.model.encode(texts, show_progress_bar=False, normalize_embeddings=True)

    def encode_single(self, text: str, is_query: bool = False) -> np.ndarray:
        return self.encode([text], is_query=is_query)[0]

    def encode_query(self, queries: List[str]) -> np.ndarray:
        return self.encode(queries, is_query=True)

    def encode_documents(self, documents: List[str]) -> np.ndarray:
        return self.encode(documents, is_query=False)
