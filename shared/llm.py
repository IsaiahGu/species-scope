"""Shared LLM client: local vLLM (OpenAI-compatible) or remote API, with a SQLite cache.

Usage:
    from llm import LLM
    llm = LLM("local")                      # Qwen3.5-9B served by vLLM on :8000, thinking off
    outs = llm.chat_many([[{"role":"user","content":"hi"}]], max_tokens=16)
    api = LLM("gpt-4o-mini")                # remote relay (OPENAI_BASE_URL / OPENAI_API_KEY)

Every call is cached by (model, messages, params); re-running a script never re-pays.
"""
from __future__ import annotations

from pathlib import Path

WORKSPACE_ROOT = Path(__file__).resolve().parents[1].as_posix()

import asyncio
import hashlib
import json
import os
import sqlite3
import threading
import time
from typing import Any

from openai import AsyncOpenAI

LOCAL_MODELS = {
    # alias -> (served model name, base_url)
    "local": ("qwen3.5-9b", os.environ.get("LOCAL_LLM_URL", "http://localhost:8000/v1")),
    "qwen3.5-9b": ("qwen3.5-9b", os.environ.get("LOCAL_LLM_URL", "http://localhost:8000/v1")),
    "gemma-4-12b": ("gemma-4-12b", os.environ.get("LOCAL_LLM2_URL", "http://localhost:8001/v1")),
}
# more local servers: LOCAL_EXTRA='{"qwen3.5-2b": ["qwen3.5-2b", "http://localhost:8002/v1"]}'
LOCAL_MODELS.update({k: tuple(v) for k, v in json.loads(os.environ.get("LOCAL_EXTRA", "{}")).items()})
CACHE_DIR = os.environ.get("LLM_CACHE_DIR", f"{WORKSPACE_ROOT}/shared/llm_cache")
USAGE_LOG = os.environ.get("LLM_USAGE_LOG", f"{WORKSPACE_ROOT}/logs/api_usage.jsonl")
# Only the API models the user listed may be called.
ALLOWED_API = {"gpt-4o-mini", "gpt-6-luna", "claude-haiku-5-5"}


class _Cache:
    def __init__(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.conn = sqlite3.connect(path, check_same_thread=False, timeout=60)
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("CREATE TABLE IF NOT EXISTS c (k TEXT PRIMARY KEY, v TEXT)")
        self.lock = threading.Lock()

    def get(self, k):
        with self.lock:
            r = self.conn.execute("SELECT v FROM c WHERE k=?", (k,)).fetchone()
        return json.loads(r[0]) if r else None

    def put_many(self, items):
        with self.lock:
            self.conn.executemany("INSERT OR REPLACE INTO c VALUES (?,?)",
                                  [(k, json.dumps(v)) for k, v in items])
            self.conn.commit()


class LLM:
    def __init__(self, model: str = "local", cache_name: str | None = None, concurrency: int | None = None):

        self.alias = model
        if model in LOCAL_MODELS:
            self.model, base = LOCAL_MODELS[model]
            self.client = AsyncOpenAI(base_url=base, api_key="EMPTY", timeout=1800, max_retries=3)
            self.local = True
            self.concurrency = concurrency or 256
        else:
            if model not in ALLOWED_API:
                raise ValueError(f"API model {model} is not on the allowed list {sorted(ALLOWED_API)}")
            self.model = model
            self.client = AsyncOpenAI(base_url=os.environ["OPENAI_BASE_URL"],
                                      api_key=os.environ["OPENAI_API_KEY"], timeout=300, max_retries=5)
            self.local = False
            self.concurrency = int(os.environ.get("API_CONC", 0)) or concurrency or 16
        name = cache_name or self.model.replace("/", "_")
        self.cache_name = name
        self.cache = _Cache(os.path.join(CACHE_DIR, f"{name}.sqlite"))

    def _key(self, messages, params):
        s = json.dumps({"m": self.model, "msgs": messages, "p": params}, sort_keys=True)
        return hashlib.sha256(s.encode()).hexdigest()

    async def _one(self, sem, messages, params):
        async with sem:
            kw = dict(model=self.model, messages=messages, **{k: v for k, v in params.items() if k != "thinking"})
            if self.local:
                kw["extra_body"] = {"chat_template_kwargs": {"enable_thinking": bool(params.get("thinking", False))}}
            # local server: wait out restarts (up to ~2 h); remote API: a few retries
            max_attempts = 400 if self.local else 8
            for attempt in range(max_attempts):
                try:
                    r = await self.client.chat.completions.create(**kw)
                    if not self.local:  # token accounting for every paid call
                        u = getattr(r, "usage", None)
                        if u is not None:
                            det = getattr(u, "completion_tokens_details", None)
                            with open(USAGE_LOG, "a") as uf:
                                uf.write(json.dumps({"t": round(time.time()), "model": self.model, "cache": self.cache_name,
                                                     "prompt": u.prompt_tokens, "completion": u.completion_tokens,
                                                     "reasoning": getattr(det, "reasoning_tokens", None) if det else None}) + "\n")
                    out = []
                    for ch in r.choices:
                        item = {"text": ch.message.content or "", "finish": ch.finish_reason}
                        if params.get("logprobs") and ch.logprobs and ch.logprobs.content:
                            item["logprobs"] = [
                                {"tok": t.token, "lp": t.logprob,
                                 "top": {x.token: x.logprob for x in (t.top_logprobs or [])}}
                                for t in ch.logprobs.content[:8]]
                        out.append(item)
                    return out
                except Exception as e:  # noqa: BLE001
                    msg = str(e)
                    # a request the server rejects as invalid will never succeed
                    if "400" in msg[:40] or "BadRequest" in type(e).__name__ or attempt == max_attempts - 1:
                        return [{"text": "", "finish": "error", "error": msg[:300]}]
                    await asyncio.sleep(min(20, 2 * (attempt + 1)))

    def chat_many(self, batch: list[list[dict]], max_tokens: int = 512, temperature: float = 0.0,
                  n: int = 1, progress: bool = True, **extra) -> list[list[dict]]:
        """Return, for each conversation, a list of n choices {text, finish, [logprobs]}."""
        params = {"max_tokens": max_tokens, "temperature": temperature, "n": n, **extra}
        if params.get("logprobs") and "top_logprobs" not in params:
            params["top_logprobs"] = 10
        keys = [self._key(m, params) for m in batch]
        out: list[Any] = [None] * len(batch)
        todo = []
        for i, (k, m) in enumerate(zip(keys, batch)):
            c = self.cache.get(k)
            if c is not None:
                out[i] = c
            else:
                todo.append((i, k, m))
        if todo:
            out_new = asyncio.run(self._run(todo, params, progress))
            for i, v in out_new.items():
                out[i] = v
        return out

    async def _run(self, todo, params, progress):
        sem = asyncio.Semaphore(self.concurrency)
        t0 = time.time()
        res = {}
        buf = []
        cnt = [0]
        step = max(1, len(todo) // 20)

        async def wrap(i, k, m):
            r = await self._one(sem, m, params)
            res[i] = r
            if r and r[0]["finish"] != "error":
                buf.append((k, r))
            if len(buf) >= 200:
                self.cache.put_many(buf[:])
                buf.clear()
            cnt[0] += 1
            if progress and cnt[0] % step == 0:
                print(f"  [{self.alias}] {cnt[0]}/{len(todo)} {time.time()-t0:.0f}s", flush=True)

        await asyncio.gather(*[wrap(i, k, m) for i, k, m in todo])
        if buf:
            self.cache.put_many(buf)
        return res

    def chat(self, messages, **kw) -> str:
        return self.chat_many([messages], progress=False, **kw)[0][0]["text"]


def parse_json(text: str):
    """Extract the first JSON object/array from a model reply."""
    t = text.strip()
    if t.startswith("```"):
        t = t.strip("`")
        t = t[t.find("\n") + 1:] if "\n" in t else t
    for opener, closer in (("{", "}"), ("[", "]")):
        s = t.find(opener)
        e = t.rfind(closer)
        if s != -1 and e > s:
            try:
                return json.loads(t[s:e + 1])
            except Exception:  # noqa: BLE001
                continue
    return None
