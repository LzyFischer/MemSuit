"""
LLM client: thin wrapper around the OpenAI-compatible API.

Handles retries, optional streaming, Qwen thinking mode, and robust JSON
extraction from raw LLM responses.
"""
import json
import re
import time
from typing import Any, Dict, List, Optional

from openai import OpenAI

import config


class LLMClient:
    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        base_url: Optional[str] = None,
        enable_thinking: Optional[bool] = None,
        use_streaming: Optional[bool] = None,
    ):
        self.api_key = api_key or config.OPENAI_API_KEY
        self.model = model or config.LLM_MODEL
        self.base_url = base_url if base_url is not None else getattr(config, "OPENAI_BASE_URL", None)
        self.enable_thinking = enable_thinking if enable_thinking is not None else getattr(config, "ENABLE_THINKING", False)
        self.use_streaming = use_streaming if use_streaming is not None else getattr(config, "USE_STREAMING", False)

        if self.base_url:
            print(f"LLM base URL: {self.base_url}")
        if self.enable_thinking:
            print("Deep thinking mode enabled")

        self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)

    # ------------------------------------------------------------------
    # Core API call
    # ------------------------------------------------------------------

    def chat_completion(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.2,
        response_format: Optional[Dict[str, str]] = None,
        max_retries: int = 3,
    ) -> str:
        kwargs: Dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
        }
        if response_format:
            kwargs["response_format"] = response_format

        # Qwen API: requires explicit enable_thinking param
        is_qwen_api = self.base_url and "dashscope.aliyuncs.com" in self.base_url
        if is_qwen_api:
            use_thinking = self.use_streaming and self.enable_thinking and not response_format
            kwargs["extra_body"] = {"enable_thinking": use_thinking}

        last_exc: Optional[Exception] = None
        for attempt in range(max_retries):
            try:
                if self.use_streaming:
                    kwargs["stream"] = True
                    return self._collect_stream(**kwargs)
                else:
                    resp = self.client.chat.completions.create(**kwargs)
                    return resp.choices[0].message.content or ""
            except Exception as e:
                last_exc = e
                if attempt < max_retries - 1:
                    wait = 2 ** attempt
                    print(f"LLM attempt {attempt + 1}/{max_retries} failed: {e}. Retry in {wait}s")
                    time.sleep(wait)
                else:
                    print(f"LLM failed after {max_retries} attempts: {e}")

        if last_exc:
            raise last_exc
        raise RuntimeError("LLM call failed")

    def _collect_stream(self, **kwargs: Any) -> str:
        chunks: List[str] = []
        for chunk in self.client.chat.completions.create(**kwargs):
            if chunk.choices and chunk.choices[0].delta.content:
                chunks.append(chunk.choices[0].delta.content)
        return "".join(chunks)

    # ------------------------------------------------------------------
    # JSON extraction
    # ------------------------------------------------------------------

    def extract_json(self, text: str) -> Any:
        """Robustly extract a JSON object/array from raw LLM output."""
        if not text or not text.strip():
            raise ValueError("Empty LLM response")

        text = text.strip()

        # Strip common preamble phrases
        for prefix in ["Here's the JSON:", "JSON:", "Result:", "Output:", "Answer:"]:
            if text.lower().startswith(prefix.lower()):
                text = text[len(prefix):].strip()

        # 1. Direct parse
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        # 2. ```json ... ``` block
        m = re.search(r"```json\s*(.*?)```", text, re.DOTALL | re.IGNORECASE)
        if m:
            try:
                return json.loads(self._clean_json(m.group(1).strip()))
            except json.JSONDecodeError:
                pass

        # 3. Generic ``` ... ``` block
        m = re.search(r"```\w*\s*(.*?)```", text, re.DOTALL)
        if m:
            try:
                return json.loads(self._clean_json(m.group(1).strip()))
            except json.JSONDecodeError:
                pass

        # 4. Balanced brace/bracket scan
        for start_char in ["{", "["]:
            result = self._balanced_extract(text, start_char)
            if result is not None:
                return result

        raise ValueError(f"Could not parse JSON from response (first 200 chars): {text[:200]}")

    @staticmethod
    def _clean_json(s: str) -> str:
        s = re.sub(r",(\s*[}\]])", r"\1", s)          # trailing commas
        s = re.sub(r"//.*?$", "", s, flags=re.MULTILINE)  # // comments
        s = re.sub(r"/\*.*?\*/", "", s, flags=re.DOTALL)  # /* */ comments
        return s.strip()

    def _balanced_extract(self, text: str, start_char: str) -> Any:
        end_char = "}" if start_char == "{" else "]"
        idx = text.find(start_char)
        if idx == -1:
            return None
        depth, in_str, esc = 0, False, False
        for i in range(idx, len(text)):
            c = text[i]
            if esc:
                esc = False
                continue
            if c == "\\":
                esc = True
                continue
            if c == '"':
                in_str = not in_str
                continue
            if in_str:
                continue
            if c == start_char:
                depth += 1
            elif c == end_char:
                depth -= 1
                if depth == 0:
                    candidate = text[idx: i + 1]
                    try:
                        return json.loads(candidate)
                    except json.JSONDecodeError:
                        try:
                            return json.loads(self._clean_json(candidate))
                        except json.JSONDecodeError:
                            return None
        return None
