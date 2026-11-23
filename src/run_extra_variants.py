"""P04 follow-up: is the weak response to "in humans" a wording problem?

Human studies rarely say "humans" in the abstract (they say patients, participants, adults), while
animal studies do say "mice". Two extra scope phrasings are run with every retriever on the same
queries, reusing the cached document embeddings: "<q> in patients", "<q> in human participants".
MODELS selects the retrievers (bm25, e5, qwen3-emb, medcpt, medcpt+ce run in the main env;
harrier, egemma2 in envs/embed5).
Output: results/runs_extra_<ds>_<tag>.parquet (same columns as runs_<ds>.parquet)
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
EXTRA = {"patients": " in patients", "participants": " in human participants"}
MODELS = os.environ.get("MODELS", "bm25,e5,qwen3-emb,medcpt,medcpt+ce").split(",")
TAG = os.environ.get("TAG", "main")

for ds in sys.argv[1:] or ["nfcorpus", "bioasq", "trec-covid"]:
    ids, titles, texts, qs, qrels = load(ds)
    qids = sorted(qs)
    full = [f"{t} {x}".strip() for t, x in zip(titles, texts)]
    runs = []
    if "bm25" in MODELS:
        import bm25s
        retr = bm25s.BM25()
        retr.index(bm25s.tokenize(full, stopwords="en", show_progress=False), show_progress=False)
        for scope, suf in EXTRA.items():
            res, sc = retr.retrieve(bm25s.tokenize([qs[q] + suf for q in qids], stopwords="en", show_progress=False),
                                    k=K, show_progress=False, n_threads=8)
            for i, q in enumerate(qids):
                for r, (j, s) in enumerate(zip(res[i], sc[i])):
                    runs.append((ds, "bm25", scope, q, ids[j], r, float(s)))
        del retr
    for m in [x for x in MODELS if x in ("e5", "qwen3-emb", "medcpt", "harrier", "egemma2")]:
        docs = list(zip(titles, texts)) if m in ("medcpt", "egemma2") else full
        De = encode(m, docs, "doc", cache_key=f"p04_{ds}")
        for scope, suf in EXTRA.items():
            qt = [qs[q] + suf for q in qids]
            I, S = search(encode(m, qt, "query"), De, k=K)
            for i, q in enumerate(qids):
                for r, (j, s) in enumerate(zip(I[i], S[i])):
                    runs.append((ds, m, scope, q, ids[j], r, float(s)))
            if m == "medcpt" and "medcpt+ce" in MODELS:  # cross-encoder rerank of the MedCPT top-100
                import torch
                from transformers import AutoModelForSequenceClassification, AutoTokenizer
                tok = AutoTokenizer.from_pretrained(f"{WORKSPACE_ROOT}/models/medcpt-cross")
                ce = AutoModelForSequenceClassification.from_pretrained(f"{WORKSPACE_ROOT}/models/medcpt-cross", torch_dtype=torch.float16).cuda().eval()
                for i, q in enumerate(qids):
                    cand = I[i]
                    scs = []
                    for s0 in range(0, len(cand), 50):
                        e = tok([[qt[i], full[j]] for j in cand[s0:s0 + 50]], truncation=True, padding=True,
                                max_length=512, return_tensors="pt").to("cuda")
                        with torch.inference_mode():
                            scs += ce(**e).logits.squeeze(-1).float().cpu().tolist()
                    for r, k_ in enumerate(np.argsort(-np.array(scs))):
                        runs.append((ds, "medcpt+ce", scope, q, ids[cand[k_]], r, float(scs[k_])))
                del ce
                torch.cuda.empty_cache()
        free()
        print(ds, m, "done", flush=True)
    pd.DataFrame(runs, columns=["ds", "retriever", "scope", "qid", "doc", "rank", "score"]).to_parquet(
        f"{ROOT}/results/runs_extra_{ds}_{TAG}.parquet")
    print(ds, "written", len(runs), flush=True)
