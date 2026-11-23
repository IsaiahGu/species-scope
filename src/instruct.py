"""P04 S4: restricting the population with instructions instead of query suffixes.

1. embedder task instructions (query side), for the instruction-following embedders (Qwen3-Embedding,
   harrier-oss-v1, EmbeddingGemma 2): "humans" instruction, the reverse "animals" instruction (to check that the
   instruction is read at all), against the default instruction of the main runs;
2. Qwen3-Reranker-0.6B over the top 100 of BM25 and of Qwen3-Embedding, with the default relevance instruction,
   the humans instruction and the animals instruction;
3. a local LLM (Qwen3.5-9B on vLLM) scoring each of the top 20 of Qwen3-Embedding with p(yes) for "relevant and
   conducted in humans" vs "relevant" (pointwise rerank).
python instruct.py emb|rerank|llm <ds> [...]   Output: results/runs_instr_<kind>_<ds>.parquet (sweep.py columns)
"""

from pathlib import Path

WORKSPACE_ROOT = Path(__file__).resolve().parents[1].as_posix()
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, f"{WORKSPACE_ROOT}/shared")
sys.path.insert(0, f"{WORKSPACE_ROOT}/src")
from embed import encode, free, search  # noqa: E402
from run_retrieval import load  # noqa: E402

ROOT = WORKSPACE_ROOT
K = 100
INSTR = {"default": "Given a web search query, retrieve relevant passages that answer the query",
         "humans": "Given a health question, retrieve studies conducted in humans that answer it; do not retrieve animal "
                   "experiments or in vitro studies",
         "animals": "Given a health question, retrieve animal experiments (studies in mice, rats or other animals) that answer it"}


def run_emb(ds):
    ids, titles, texts, qs, qrels = load(ds)
    qids = sorted(qs)
    full = [f"{t} {x}".strip() for t, x in zip(titles, texts)]
    runs = []
    for m in ("qwen3-emb", "harrier", "egemma2"):
        docs = list(zip(titles, texts)) if m == "egemma2" else full
        De = encode(m, docs, "doc", cache_key=f"p04_{ds}")
        for name, ins in INSTR.items():
            if m == "egemma2":
                if name == "default":
                    Qe = encode(m, [qs[q] for q in qids], "query")
                else:  # EmbeddingGemma takes the task description in its prompt prefix
                    from embed import _get
                    st = _get("egemma2")
                    Qe = np.asarray(st.encode([f"task: {ins} | query: {qs[q]}" for q in qids], batch_size=64,
                                              normalize_embeddings=True, show_progress_bar=False), dtype=np.float16)
            else:
                Qe = encode(m, [qs[q] for q in qids], "query", qwen_task=ins)
            I, S = search(Qe, De, k=K)
            for i, q in enumerate(qids):
                runs += [(ds, m, f"instr_{name}", q, ids[j], r, float(s)) for r, (j, s) in enumerate(zip(I[i], S[i]))]
        free()
        print(ds, m, "done", flush=True)
    return runs


def run_rerank(ds):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    ids, titles, texts, qs, qrels = load(ds)
    full = {d: f"{t} {x}".strip() for d, t, x in zip(ids, titles, texts)}
    S = pd.read_parquet(f"{ROOT}/results/runs_sweep_{ds}.parquet")
    S = S[(S.scope == "base") & S.retriever.isin(["bm25", "qwen3-emb"])]
    path = f"{WORKSPACE_ROOT}/models/qwen3-reranker-0.6b"
    tok = AutoTokenizer.from_pretrained(path, padding_side="left")
    mod = AutoModelForCausalLM.from_pretrained(path, torch_dtype=torch.float16).cuda().eval()
    yes, no = tok.convert_tokens_to_ids("yes"), tok.convert_tokens_to_ids("no")
    pre = ("<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and the Instruct provided. "
           "Note that the answer can only be \"yes\" or \"no\".<|im_end|>\n<|im_start|>user\n")
    suf = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
    pre_ids, suf_ids = tok.encode(pre, add_special_tokens=False), tok.encode(suf, add_special_tokens=False)
    MAXLEN, BUDGET = 640, 40000  # tokens per sequence, tokens per batch
    groups = [(first, q, g.sort_values("rank").doc.tolist()) for (first, q), g in S.groupby(["retriever", "qid"])]
    items = [(gi, name, d) for gi, (_, _, docs) in enumerate(groups) for name in INSTR for d in docs]
    body = tok([f"<Instruct>: {INSTR[name]}\n<Query>: {qs[groups[gi][1]]}\n<Document>: {full[d][:3000]}" for gi, name, d in items],
               add_special_tokens=False, truncation=True, max_length=MAXLEN - len(pre_ids) - len(suf_ids))["input_ids"]
    seqs = [pre_ids + x + suf_ids for x in body]
    order = np.argsort([len(x) for x in seqs], kind="stable")
    sc = np.zeros(len(seqs))
    i = 0
    with torch.inference_mode():
        while i < len(order):
            j = i + 1
            while j < len(order) and (j - i + 1) * len(seqs[order[j]]) <= BUDGET and j - i < 512:
                j += 1
            idx = order[i:j]
            e = tok.pad({"input_ids": [seqs[k] for k in idx]}, padding=True, return_tensors="pt").to("cuda")
            lg = mod(**e, logits_to_keep=1).logits[:, -1, :]
            sc[idx] = torch.log_softmax(torch.stack([lg[:, no], lg[:, yes]], 1).float(), 1)[:, 1].exp().cpu().numpy()
            i = j
            if (i // 20000) != ((i - len(idx)) // 20000):
                print(ds, "rerank", i, "/", len(order), flush=True)
    runs, k = [], 0
    for first, q, docs in groups:
        for name in INSTR:
            s_ = sc[k:k + len(docs)]; k += len(docs)
            for r, j in enumerate(np.argsort(-s_, kind="stable")):
                runs.append((ds, f"qrerank@{first}", f"instr_{name}", q, docs[j], r, float(s_[j])))
    return runs


def run_llm(ds):
    sys.path.insert(0, f"{WORKSPACE_ROOT}/shared")
    from llm import LLM
    ids, titles, texts, qs, qrels = load(ds)
    full = {d: f"{t} {x}".strip() for d, t, x in zip(ids, titles, texts)}
    S = pd.read_parquet(f"{ROOT}/results/runs_sweep_{ds}.parquet")
    S = S[(S.scope == "base") & (S.retriever == "qwen3-emb") & (S["rank"] < 20)]
    P = {"relevant": "Query: {q}\n\nDocument: {d}\n\nIs this document relevant to the query? Answer yes or no.",
         "humans": "Query: {q}\n\nDocument: {d}\n\nIs this document relevant to the query and is it a study conducted in "
                   "humans (not an animal experiment or an in vitro study)? Answer yes or no."}
    llm = LLM("local", cache_name="p04_llmrerank")
    runs = []
    for name, tmpl in P.items():
        keys = list(S[["qid", "doc", "rank"]].itertuples(index=False))
        outs = llm.chat_many([[{"role": "user", "content": tmpl.format(q=qs[q], d=full[d][:3000])}] for q, d, _ in keys],
                             max_tokens=2, logprobs=True, top_logprobs=10)
        sc = {}
        for (q, d, r0), o in zip(keys, outs):
            top = (o[0].get("logprobs") or [{"top": {}}])[0]["top"]
            y = max([v for t, v in top.items() if t.strip().lower().startswith("yes")], default=-50)
            n = max([v for t, v in top.items() if t.strip().lower().startswith("no")], default=-50)
            sc.setdefault(q, []).append((d, float(np.exp(y) / (np.exp(y) + np.exp(n))), r0))
        for q, lst in sc.items():
            lst.sort(key=lambda x: (-x[1], x[2]))
            runs += [(ds, "llmrerank@qwen3-emb", f"llm_{name}", q, d, r, s) for r, (d, s, _) in enumerate(lst)]
    return runs


if __name__ == "__main__":
    kind = sys.argv[1]
    for ds in sys.argv[2:]:
        runs = {"emb": run_emb, "rerank": run_rerank, "llm": run_llm}[kind](ds)
        pd.DataFrame(runs, columns=["ds", "retriever", "scope", "qid", "doc", "rank", "score"]).to_parquet(
            f"{ROOT}/results/runs_instr_{kind}_{ds}.parquet")
        print(ds, kind, "written", len(runs), flush=True)
