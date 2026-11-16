from typing import Any, Dict, List, Optional, Sequence
import os, json, time, math, re
from dataclasses import dataclass

try:
    from retry.api import retry_call
except Exception:
    retry_call = None

from utils import set_seed

set_seed(42)


@dataclass
class LLMClient:
    def __init__(self, api_key: str):
        if not api_key:
            raise RuntimeError("API key is required.")
        from together import Together
        self.api_key = api_key
        self.client = Together(api_key=api_key)
        self.chat_model = "openai/gpt-oss-20b"
        self.embed_model = "togethercomputer/m2-bert-80M-32k-retrieval"
        self.request_timeout_s = 60
            
    
    def embed_batch(self, texts: Sequence[str]) -> List[List[float]]:
        """Return L2-normalized embeddings for a batch of texts."""
        def call():
            return self.client.embeddings.create(model=self.embed_model, input=list(texts))
        resp = self._with_retry(call)
        vecs = [d.embedding for d in resp.data]
        # L2 normalize
        out = []
        for v in vecs:
            norm = math.sqrt(sum(x * x for x in v)) or 1.0
            out.append([x / norm for x in v])
        return out

    def embed_one(self, text: str) -> List[float]:
        vecs = self.embed_batch([text])
        return vecs[0] if vecs else []

    async def embed_one_async(self, text: str) -> List[float]:
        vecs = await self.embed_many_async([text], batch_size=1, concurrency=1)
        return vecs[0] if vecs else []

    async def embed_many_async(self, texts: Sequence[str], batch_size: int = 16, concurrency: int = 4) -> List[List[float]]:
        """Asynchronously embed many texts by batching and running batches concurrently.

        - Uses embed_batch under the hood in worker threads to avoid blocking the loop.
        - Applies L2 normalization from embed_batch.
        """
        import asyncio
        from asyncio import Semaphore

        if not texts:
            return []

        # Prepare batched jobs
        batches = [(i, list(texts[i:i + batch_size])) for i in range(0, len(texts), batch_size)]
        results: List[Optional[List[float]]] = [None] * len(texts)  # type: ignore
        sem = Semaphore(max(1, concurrency))

        async def worker(start_index: int, batch_inputs: Sequence[str]) -> None:
            async with sem:
                vecs = await asyncio.to_thread(self.embed_batch, batch_inputs)
                for offset, v in enumerate(vecs):
                    results[start_index + offset] = v

        await asyncio.gather(*[worker(start, batch) for (start, batch) in batches])
        # type: ignore
        return [v for v in results if v is not None]

    def chat(self, messages, *, max_tokens=400, temperature=0.5, extra=None) -> str:
        def call():
            return self.client.chat.completions.create(
                model=self.chat_model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                **(extra or {}),
            )
        resp = self._with_retry(call)
        return resp.choices[0].message.content
    
    async def chat_async(self, messages, *, max_tokens=400, temperature=0.2, extra=None) -> str:
        import asyncio
        return await asyncio.to_thread(self.chat, messages, max_tokens=max_tokens, temperature=temperature, extra=extra)

    def chat_json(self, system: str, user: str, *, schema_hint: Optional[str] = None,
                    max_tokens: int = 400, temperature: float = 0.0, max_retry: int = 3) -> Dict[str, Any]:
        system = system.strip()
        if schema_hint:
            system += "\n\nReturn the queries as a JSON. JSON schema (hint):\n" + schema_hint

        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        extra = {"response_format": {"type": "json_object"}}

        for retry in range(max_retry):
            txt = self.chat(messages, max_tokens=max_tokens, temperature=temperature, extra=extra)
            # strict parse -> fallback rescue
            try:
                return json.loads(txt)
            except Exception:
                m = re.search(r"\{(?:.|\n)*\}", txt)
                if m:
                    try:
                        return json.loads(m.group(0))
                    except Exception:
                        pass
            if retry < max_retry - 1:
                time.sleep(1)

        return {"_error": "invalid_json_from_model", "_raw": txt}
    
    def _with_retry(self, fn):
        if retry_call is not None:
            return retry_call(fn, tries=4, delay=1, backoff=2, max_delay=10, exceptions=Exception, logger=None)
        delay = 1.0
        for attempt in range(4):
            try:
                return fn()
            except Exception:
                if attempt == 3:
                    raise
                time.sleep(delay)
                delay = min(10.0, delay * 2.0)