"""
Hybrid Retriever — Stage 3: Intent-Aware Retrieval Planning (§3.3)

Pipeline per query (semantic-only):
  1. Analyze information requirements → retrieval plan
  2. Generate targeted semantic queries
  3. Execute multi-query semantic retrieval (parallel or sequential)
  4. Merge & deduplicate
  5. (Optional) Reflection: re-query for missing information

The lexical (BM25) and symbolic (metadata-filter) retrieval layers have been
removed — retrieval is now purely embedding-similarity over the
`lossless_restatement` field of stored entries.

The class is still called `HybridRetriever` for backwards compatibility with
the rest of the system (`SimpleMemSystem`, eval, etc.); "hybrid" now refers
to plan + multi-query + reflection rather than to multiple index layers.
"""
import concurrent.futures
from typing import Any, Dict, List, Optional

import config
from database.vector_store import VectorStore
from models.memory_entry import MemoryEntry
from utils.llm_client import LLMClient


class HybridRetriever:
    def __init__(
        self,
        llm_client: LLMClient,
        vector_store: VectorStore,
        semantic_top_k: Optional[int] = None,
        enable_planning: Optional[bool] = None,
        enable_reflection: Optional[bool] = None,
        max_reflection_rounds: Optional[int] = None,
        enable_parallel_retrieval: Optional[bool] = None,
        max_retrieval_workers: Optional[int] = None,
    ):
        self.llm_client = llm_client
        self.vector_store = vector_store
        self.semantic_top_k = semantic_top_k or config.SEMANTIC_TOP_K

        self.enable_planning = (
            enable_planning if enable_planning is not None
            else getattr(config, "ENABLE_PLANNING", True)
        )
        self.enable_reflection = (
            enable_reflection if enable_reflection is not None
            else getattr(config, "ENABLE_REFLECTION", True)
        )
        self.max_reflection_rounds = (
            max_reflection_rounds if max_reflection_rounds is not None
            else getattr(config, "MAX_REFLECTION_ROUNDS", 2)
        )
        self.enable_parallel_retrieval = (
            enable_parallel_retrieval if enable_parallel_retrieval is not None
            else getattr(config, "ENABLE_PARALLEL_RETRIEVAL", True)
        )
        self.max_retrieval_workers = (
            max_retrieval_workers if max_retrieval_workers is not None
            else getattr(config, "MAX_RETRIEVAL_WORKERS", 3)
        )

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def retrieve(
        self, query: str, enable_reflection: Optional[bool] = None
    ) -> List[MemoryEntry]:
        """
        Main retrieval entry point.

        Args:
            query: User question.
            enable_reflection: Per-call override of the global reflection flag
                               (set False for adversarial / category-5 questions).
        """
        if self.enable_planning:
            return self._retrieve_with_planning(query, enable_reflection)
        return self._semantic_search(query)

    # ------------------------------------------------------------------
    # Stage 3 — Intent-aware retrieval with planning
    # ------------------------------------------------------------------

    def _retrieve_with_planning(
        self, query: str, enable_reflection: Optional[bool] = None
    ) -> List[MemoryEntry]:
        print(f"\n[Planning] Query: {query}")

        # Step 1 — analyze information requirements
        plan = self._analyze_requirements(query)
        print(f"[Planning] {len(plan.get('required_info', []))} requirements identified")

        # Step 2 — generate targeted semantic queries
        queries = self._generate_queries(query, plan)
        print(f"[Planning] {len(queries)} targeted queries")

        # Step 3 — semantic search (parallel or sequential)
        if self.enable_parallel_retrieval and len(queries) > 1:
            results = self._parallel_semantic_search(queries)
        else:
            results = []
            for i, q in enumerate(queries, 1):
                print(f"[Search {i}] {q}")
                results.extend(self._semantic_search(q))

        # Step 4 — merge
        merged = self._deduplicate(results)
        print(f"[Planning] {len(merged)} unique results after merge")

        # Step 5 — optional reflection
        use_reflection = (
            enable_reflection if enable_reflection is not None else self.enable_reflection
        )
        if use_reflection:
            merged = self._reflection_loop(query, merged, plan)

        return merged

    # ------------------------------------------------------------------
    # Semantic retrieval
    # ------------------------------------------------------------------

    def _semantic_search(self, query: str) -> List[MemoryEntry]:
        return self.vector_store.semantic_search(query, top_k=self.semantic_top_k)

    def _parallel_semantic_search(self, queries: List[str]) -> List[MemoryEntry]:
        results: List[MemoryEntry] = []
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=self.max_retrieval_workers
        ) as ex:
            futures = {ex.submit(self._semantic_search, q): i for i, q in enumerate(queries, 1)}
            for future in concurrent.futures.as_completed(futures):
                idx = futures[future]
                try:
                    batch = future.result()
                    results.extend(batch)
                    print(f"[Search {idx}] {len(batch)} results")
                except Exception as e:
                    print(f"[Search {idx}] failed: {e}")
        return results

    def _parallel_extra_search(self, queries: List[str]) -> List[MemoryEntry]:
        results: List[MemoryEntry] = []
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=self.max_retrieval_workers
        ) as ex:
            futures = {ex.submit(self._semantic_search, q): i for i, q in enumerate(queries, 1)}
            for future in concurrent.futures.as_completed(futures):
                idx = futures[future]
                try:
                    results.extend(future.result())
                except Exception as e:
                    print(f"[Extra search {idx}] failed: {e}")
        return results

    # ------------------------------------------------------------------
    # Reflection loop
    # ------------------------------------------------------------------

    def _reflection_loop(
        self,
        query: str,
        results: List[MemoryEntry],
        plan: Dict[str, Any],
    ) -> List[MemoryEntry]:
        for rnd in range(self.max_reflection_rounds):
            print(f"\n[Reflection {rnd + 1}] Checking completeness...")
            status = self._check_completeness(query, results, plan)
            if status == "complete":
                print(f"[Reflection {rnd + 1}] Complete ✓")
                break
            if status == "no_results":
                print(f"[Reflection {rnd + 1}] No results, stopping")
                break
            # incomplete — generate follow-up queries
            extra_queries = self._missing_info_queries(query, results, plan)
            print(f"[Reflection {rnd + 1}] {len(extra_queries)} follow-up queries")
            extra = (
                self._parallel_extra_search(extra_queries)
                if self.enable_parallel_retrieval and len(extra_queries) > 1
                else [r for q in extra_queries for r in self._semantic_search(q)]
            )
            results = self._deduplicate(results + extra)
            print(f"[Reflection {rnd + 1}] {len(results)} total results")
        return results

    # ------------------------------------------------------------------
    # LLM helpers — query analysis & planning
    # ------------------------------------------------------------------

    def _analyze_requirements(self, query: str) -> Dict[str, Any]:
        prompt = f"""
Analyze the following question and determine what specific information is required to answer it comprehensively.

Question: {query}

Think step by step:
1. What type of question is this? (factual, temporal, relational, explanatory, etc.)
2. What key entities, events, or concepts need to be identified?
3. What relationships or connections need to be established?
4. What minimal set of information pieces would be sufficient to answer this question?

Return your analysis in JSON format:
```json
{{
  "question_type": "type of question",
  "key_entities": ["entity1", "entity2", ...],
  "required_info": [
    {{
      "info_type": "what kind of information",
      "description": "specific information needed",
      "priority": "high/medium/low"
    }}
  ],
  "relationships": ["relationship1", "relationship2", ...],
  "minimal_queries_needed": 2
}}
```

Focus on identifying the minimal essential information needed, not exhaustive details.

Return ONLY the JSON, no other text.
"""
        return self._llm_json(prompt, default={
            "question_type": "general",
            "key_entities": [query],
            "required_info": [{"info_type": "general", "description": "relevant info", "priority": "high"}],
            "minimal_queries_needed": 1,
        })

    def _generate_queries(self, query: str, plan: Dict[str, Any]) -> List[str]:
        prompt = f"""
Based on the information requirements analysis, generate the minimal set of targeted search queries needed to gather the required information.

Original Question: {query}

Information Requirements Analysis:
- Question Type: {plan.get('question_type', 'general')}
- Key Entities: {plan.get('key_entities', [])}
- Required Information: {plan.get('required_info', [])}
- Relationships: {plan.get('relationships', [])}
- Minimal Queries Needed: {plan.get('minimal_queries_needed', 1)}

Generate the minimal set of search queries that would efficiently gather all the required information. Each query should be focused and specific to retrieve distinct types of information.

Guidelines:
1. Always include the original query as one option
2. Generate only the minimal necessary queries (usually 1-3)
3. Each query should target a specific information requirement
4. Avoid redundant or overlapping queries
5. Focus on efficiency - fewer, more targeted queries are better

Return your response in JSON format:
```json
{{
  "reasoning": "Brief explanation of the query strategy",
  "queries": [
    "targeted query 1",
    "targeted query 2",
    ...
  ]
}}
```

Return ONLY the JSON, no other text.
"""
        result = self._llm_json(prompt, default={"queries": [query]})
        queries = result.get("queries", [query])
        if query not in queries:
            queries.insert(0, query)
        return queries[:4]

    def _check_completeness(
        self, query: str, results: List[MemoryEntry], plan: Dict[str, Any]
    ) -> str:
        if not results:
            return "no_results"
        context = self._format_for_check(results)
        prompt = f"""
Analyze whether the provided information is sufficient to completely answer the original question, based on the identified information requirements.

Original Question: {query}

Required Information Types: {plan.get('required_info', [])}

Current Available Information:
{context}

Evaluate whether:
1. All required information types are addressed
2. The information is complete enough to provide a comprehensive answer
3. Any critical gaps remain that would prevent a satisfactory answer

Return your evaluation in JSON format:
```json
{{
  "assessment": "complete" OR "incomplete",
  "reasoning": "Brief explanation of completeness assessment",
  "missing_info_types": ["list", "of", "missing", "information", "types"],
  "coverage_percentage": 85
}}
```

Return ONLY the JSON, no other text.
"""
        result = self._llm_json(prompt, default={"assessment": "incomplete"})
        return result.get("assessment", "incomplete")

    def _missing_info_queries(
        self, query: str, results: List[MemoryEntry], plan: Dict[str, Any]
    ) -> List[str]:
        context = self._format_for_check(results)
        prompt = f"""
Based on the original question, required information types, and currently available information, generate targeted search queries to find the missing information needed to answer the question completely.

Original Question: {query}

Required Information Types: {plan.get('required_info', [])}

Currently Available Information:
{context}

Generate 1-3 specific search queries that would help find the missing information. Focus on:
1. Information gaps identified in the current context
2. Specific missing details needed to answer the original question
3. Different search angles that might retrieve the missing information

Return your response in JSON format:
```json
{{
  "missing_analysis": "Brief analysis of what specific information is missing",
  "targeted_queries": [
    "specific query 1 for missing info",
    "specific query 2 for missing info",
    ...
  ]
}}
```

Return ONLY the JSON, no other text.
"""
        result = self._llm_json(prompt, default={"targeted_queries": []})
        return result.get("targeted_queries", [])

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def _llm_json(self, prompt: str, default: Any) -> Any:
        messages = [
            {"role": "system", "content": "You are a missing information query generator. You must output valid JSON format."},
            {"role": "user", "content": prompt},
        ]
        response_format = (
            {"type": "json_object"} if getattr(config, "USE_JSON_FORMAT", False) else None
        )
        for attempt in range(3):
            try:
                resp = self.llm_client.chat_completion(
                    messages, temperature=0.2, response_format=response_format
                )
                return self.llm_client.extract_json(resp)
            except Exception as e:
                if attempt < 2:
                    print(f"  LLM JSON attempt {attempt + 1}/3 failed: {e}")
                else:
                    print(f"  LLM JSON failed: {e} — using default")
                    return default
        return default

    @staticmethod
    def _deduplicate(entries: List[MemoryEntry]) -> List[MemoryEntry]:
        seen, out = set(), []
        for e in entries:
            if e.entry_id not in seen:
                seen.add(e.entry_id)
                out.append(e)
        return out

    @staticmethod
    def _format_for_check(entries: List[MemoryEntry]) -> str:
        lines = []
        for i, e in enumerate(entries, 1):
            lines.append(f"[{i}] {e.lossless_restatement}")
        return "\n".join(lines)
