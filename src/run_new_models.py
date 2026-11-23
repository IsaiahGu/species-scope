"""P04: the two newest general embedders (harrier-oss-v1-0.6b, 2026-03; EmbeddingGemma 2, 2026-09)
on the same datasets and scope variants as run_retrieval.py. Runs in envs/embed5 (transformers 5).
Output: results/runs_new_<ds>.parquet, results/relscores_new_<ds>.parquet
"""

from pathlib import Path

WORKSPACE_ROOT = Path(__file__).resolve().parents[1].as_posix()
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, f"{WORKSPACE_ROOT}/shared")
sys.path.insert(0, f"{WORKSPACE_ROOT}/src")
from embed import encode, free, search  # noqa: E402
from run_retrieval import SCOPES, load  # noqa: E402

ROOT = WORKSPACE_ROOT
K = 100
NEW = ["harrier", "egemma2"]

for ds in sys.argv[1:] or ["nfcorpus", "bioasq", "trec-covid"]:
    ids, titles, texts, qs, qrels = load(ds)
    qids = sorted(qs)
    pos = {d: i for i, d in enumerate(ids)}
    runs, rel = [], []
    for m in NEW:
        docs = list(zip(titles, texts)) if m == "egemma2" else [f"{t} {x}".strip() for t, x in zip(titles, texts)]
        De = encode(m, docs, "doc", cache_key=f"p04_{ds}")
        for scope, suf in SCOPES.items():
            Qe = encode(m, [qs[q] + suf for q in qids], "query")
            I, S = search(Qe, De, k=K)
            for i, q in enumerate(qids):
                for r, (j, s) in enumerate(zip(I[i], S[i])):
                    runs.append((ds, m, scope, q, ids[j], r, float(s)))
                if scope == "base":
                    js = [pos[d] for d, g in qrels[q].items() if g > 0 and d in pos]
                    if js:
                        sc = De[js].astype(np.float32) @ Qe[i].astype(np.float32)
                        for k_, j in enumerate(js):
                            rel.append({"ds": ds, "retriever": m, "qid": q, "doc": ids[j], "grade": qrels[q][ids[j]],
                                        "score": float(sc[k_]), "len": len(f"{titles[j]} {texts[j]}".split()), "bm25": np.nan})
        free()
        print(ds, m, "done", flush=True)
    pd.DataFrame(runs, columns=["ds", "retriever", "scope", "qid", "doc", "rank", "score"]).to_parquet(f"{ROOT}/results/runs_new_{ds}.parquet")
    pd.DataFrame(rel).to_parquet(f"{ROOT}/results/relscores_new_{ds}.parquet")
