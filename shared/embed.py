"""Shared dense encoders with on-disk caching.

encode(model, texts, kind, cache_key=None) -> np.ndarray (L2-normalised, float16)
  model in: qwen3-emb, e5, bge, specter, specter2, scincl, medcpt
  kind  in: "query" | "doc"   (docs may be (title, text) tuples for scientific models)
Cache: shared/emb_cache/<model>/<cache_key>_<kind>.npy (+ .n with the row count)
"""
from __future__ import annotations

from pathlib import Path

WORKSPACE_ROOT = Path(__file__).resolve().parents[1].as_posix()

import os

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import numpy as np
import torch

M = f"{WORKSPACE_ROOT}/models"
CACHE = f"{WORKSPACE_ROOT}/shared/emb_cache"
QWEN_TASK = "Given a web search query, retrieve relevant passages that answer the query"
_loaded = {}
if torch.cuda.is_available():  # share the GPU politely with vLLM and Whisper
    torch.cuda.set_per_process_memory_fraction(float(os.environ.get("GPU_FRAC", "0.15")))

_lockf = None


def acquire_gpu():
    """Serialise GPU helper jobs: block until this process holds shared/gpu.lock.
    The lock is held until release_gpu() or process exit."""
    global _lockf
    # NO_GPU_LOCK=1 or the file shared/no_gpu_lock turns the lock off (when GPU memory is ample)
    if _lockf is None and os.environ.get("NO_GPU_LOCK") != "1" and not os.path.exists(f"{WORKSPACE_ROOT}/shared/no_gpu_lock"):
        import fcntl
        import time
        _lockf = open(f"{WORKSPACE_ROOT}/shared/gpu.lock", "w")
        t0 = time.time()
        fcntl.flock(_lockf, fcntl.LOCK_EX)
        if time.time() - t0 > 5:
            print(f"[embed] waited {time.time() - t0:.0f}s for the GPU lock", flush=True)


def release_gpu():
    global _lockf
    free()
    if _lockf is not None:
        import fcntl
        fcntl.flock(_lockf, fcntl.LOCK_UN)
        _lockf.close()
        _lockf = None


def _st(path, **kw):
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(path, device="cuda", model_kwargs={"torch_dtype": torch.float16}, **kw)


def _hf(path):
    from transformers import AutoModel, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(path)
    mod = AutoModel.from_pretrained(path, torch_dtype=torch.float16).cuda().eval()
    return tok, mod


def _get(name):
    if name in _loaded:
        return _loaded[name]
    acquire_gpu()
    if name == "qwen3-emb":
        m = _st(f"{M}/qwen3-emb-0.6b")
        m.max_seq_length = 512
    elif name == "e5":
        m = _st(f"{M}/e5-base-v2"); m.max_seq_length = 512
    elif name == "bge":
        m = _st(f"{M}/bge-base-en-v1.5"); m.max_seq_length = 512
    elif name in ("specter", "scincl"):
        m = _hf(f"{M}/{'specter' if name == 'specter' else 'scincl'}")
    elif name == "specter2":
        from adapters import AutoAdapterModel
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(f"{M}/specter2_base")
        mod = AutoAdapterModel.from_pretrained(f"{M}/specter2_base")
        mod.load_adapter(f"{M}/specter2_adapter", load_as="proximity", set_active=False)
        mod.load_adapter(f"{M}/specter2_adhoc_query", load_as="adhoc_query", set_active=False)
        mod = mod.half().cuda().eval()
        m = (tok, mod)
    elif name == "medcpt":
        m = {"query": _hf(f"{M}/medcpt-query"), "doc": _hf(f"{M}/medcpt-article")}
    elif name == "harrier":  # microsoft/harrier-oss-v1-0.6b (2026-03), Qwen3-based
        m = _st(f"{M}/harrier-oss-v1-0.6b"); m.max_seq_length = 512
    elif name == "egemma2":  # google/embeddinggemma-2 (2026-09); needs transformers 5 (envs/embed5), bf16
        from sentence_transformers import SentenceTransformer
        m = SentenceTransformer(f"{M}/embeddinggemma-2", device="cuda", model_kwargs={"torch_dtype": torch.bfloat16})
        m.max_seq_length = 512
    else:
        raise ValueError(name)
    _loaded[name] = m
    return m


def _cls_encode(tok, mod, texts, bs=64, max_len=512, pairs=False):
    out = []
    for s in range(0, len(texts), bs):
        chunk = texts[s:s + bs]
        if pairs:
            enc = tok([a for a, _ in chunk], [b for _, b in chunk], padding=True, truncation="longest_first",
                      max_length=max_len, return_tensors="pt")
        else:
            enc = tok(chunk, padding=True, truncation=True, max_length=max_len, return_tensors="pt")
        enc = {k: v.cuda() for k, v in enc.items()}
        with torch.inference_mode():
            h = mod(**enc).last_hidden_state[:, 0]
        out.append(torch.nn.functional.normalize(h.float(), dim=-1).cpu().numpy())
    return np.concatenate(out) if out else np.zeros((0, 768), np.float32)


def _join(t):
    if isinstance(t, (tuple, list)):
        return (t[0] or "") + " " + (t[1] or "")
    return t


def encode(name, texts, kind="doc", cache_key=None, bs=32, qwen_task=QWEN_TASK):
    if cache_key:
        p = f"{CACHE}/{name}/{cache_key}_{kind}.npy"
        if os.path.exists(p):
            arr = np.load(p)
            if len(arr) == len(texts):
                return arr
    m = _get(name)
    if name == "qwen3-emb":
        xs = [f"Instruct: {qwen_task}\nQuery: {_join(t)}" if kind == "query" else _join(t) for t in texts]
        E = m.encode(xs, batch_size=bs, normalize_embeddings=True, show_progress_bar=False)
    elif name == "e5":
        xs = [("query: " if kind == "query" else "passage: ") + _join(t) for t in texts]
        E = m.encode(xs, batch_size=bs, normalize_embeddings=True, show_progress_bar=False)
    elif name == "bge":
        xs = [("Represent this sentence for searching relevant passages: " if kind == "query" else "") + _join(t) for t in texts]
        E = m.encode(xs, batch_size=bs, normalize_embeddings=True, show_progress_bar=False)
    elif name in ("specter", "scincl"):
        tok, mod = m
        xs = [t if isinstance(t, str) else (t[0] or "") + tok.sep_token + (t[1] or "") for t in texts]
        E = _cls_encode(tok, mod, xs, bs=32)
    elif name == "specter2":
        tok, mod = m
        mod.set_active_adapters("adhoc_query" if kind == "query" else "proximity")
        xs = [t if isinstance(t, str) else (t[0] or "") + tok.sep_token + (t[1] or "") for t in texts]
        E = _cls_encode(tok, mod, xs, bs=32)
    elif name == "harrier":
        xs = [f"Instruct: {qwen_task}\nQuery: {_join(t)}" if kind == "query" else _join(t) for t in texts]
        E = m.encode(xs, batch_size=bs, normalize_embeddings=True, show_progress_bar=False)
    elif name == "egemma2":
        if kind == "query":
            xs = ["task: search result | query: " + _join(t) for t in texts]
        else:
            xs = [f"title: {(t[0] or 'none')} | text: {t[1] or ''}" if isinstance(t, (tuple, list)) else f"title: none | text: {t}"
                  for t in texts]
        E = m.encode(xs, batch_size=bs, normalize_embeddings=True, show_progress_bar=False)
    elif name == "medcpt":
        tok, mod = m[kind]
        if kind == "query":
            E = _cls_encode(tok, mod, [_join(t) for t in texts], bs=64, max_len=64)
        else:
            xs = [(t[0] or "", t[1] or "") if isinstance(t, (tuple, list)) else ("", t) for t in texts]
            E = _cls_encode(tok, mod, xs, bs=32, max_len=512, pairs=True)
    E = np.asarray(E, dtype=np.float16)
    if cache_key:
        os.makedirs(f"{CACHE}/{name}", exist_ok=True)
        np.save(f"{CACHE}/{name}/{cache_key}_{kind}.npy", E)
    return E


def search(Q, D, k=100, bs=256):
    """Exact inner-product top-k on GPU. Returns (idx[nq,k], scores[nq,k])."""
    acquire_gpu()
    Dt = torch.from_numpy(np.ascontiguousarray(D)).cuda().half()
    I, S = [], []
    for s in range(0, len(Q), bs):
        q = torch.from_numpy(np.ascontiguousarray(Q[s:s + bs])).cuda().half()
        sc = q @ Dt.T
        v, i = sc.float().topk(min(k, D.shape[0]), dim=1)
        I.append(i.cpu().numpy()); S.append(v.cpu().numpy())
    del Dt
    return np.concatenate(I), np.concatenate(S)


def free():
    _loaded.clear()
    torch.cuda.empty_cache()
