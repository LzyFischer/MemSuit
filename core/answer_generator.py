"""
Answer Generator — final synthesis from retrieved context C_q (§3.3).
"""
from typing import List

import config
from models.memory_entry import MemoryEntry
from utils.llm_client import LLMClient


class AnswerGenerator:
    def __init__(self, llm_client: LLMClient):
        self.llm_client = llm_client

    def generate_answer(self, query: str, contexts: List[MemoryEntry]) -> str:
        if not contexts:
            return "No relevant information found"

        context_str = self._format_contexts(contexts)
        prompt = f"""
Answer the user's question based on the provided context.

User Question: {query}

Relevant Context:
{context_str}

Requirements:
1. First, think through the reasoning process
2. Then provide a very CONCISE answer (short phrase about core information)
3. Answer must be based ONLY on the provided context
4. All dates in the response must be formatted as 'DD Month YYYY' but you can output more or less details if needed
5. Return your response in JSON format

Output Format:
```json
{{
  "reasoning": "Brief explanation of your thought process",
  "answer": "Concise answer in a short phrase"
}}
```

Example:
Question: "When will they meet?"
Context: "Alice suggested meeting Bob at 2025-11-16T14:00:00..."

Output:
```json
{{
  "reasoning": "The context explicitly states the meeting time as 2025-11-16T14:00:00",
  "answer": "16 November 2025 at 2:00 PM"
}}
```

Now answer the question. Return ONLY the JSON, no other text.
"""

        messages = [
            {
                "role": "system",
                "content": "You are a Q&A assistant. Extract concise answers from context. Output valid JSON only.",
            },
            {"role": "user", "content": prompt},
        ]
        response_format = (
            {"type": "json_object"} if getattr(config, "USE_JSON_FORMAT", False) else None
        )

        for attempt in range(3):
            try:
                response = self.llm_client.chat_completion(
                    messages, temperature=0.1, response_format=response_format
                )
                result = self.llm_client.extract_json(response)
                return result.get("answer", response.strip())
            except Exception as e:
                if attempt < 2:
                    print(f"  Answer gen attempt {attempt + 1}/3 failed: {e}")
                else:
                    print(f"  Answer gen failed: {e}")
                    return "Failed to generate answer"
        return "Failed to generate answer"

    def _format_contexts(self, contexts: List[MemoryEntry]) -> str:
        parts = []
        for i, e in enumerate(contexts, 1):
            parts.append(f"[Context {i}]\nContent: {e.lossless_restatement}")
        return "\n\n".join(parts)
