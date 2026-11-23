"""P04 S1/S2/S3: suffix sweep over seven retrievers and three corpora.

Suffixes (appended to every query): S1 human-pointing, animal-pointing, in-vitro and neutral phrases;
S2 other population axes (age, sex, pregnancy, study type). base = the query alone.
Retrievers: BM25, e5-base-v2, Qwen3-Embedding-0.6B, MedCPT bi-encoder, MedCPT cross-encoder over the
MedCPT top 100, harrier-oss-v1-0.6b, EmbeddingGemma 2 (document embeddings cached in shared/emb_cache).
python sweep.py <ds> [<ds> ...]   Output: results/runs_sweep_<ds>.parquet (ds, retriever, scope, qid, doc, rank, score)
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
S1 = {"humans": " in humans", "people": " in people", "patients": " in patients", "adults": " in adults",
      "participants": " in human participants", "subjects": " in human subjects", "volunteers": " in volunteers",
      "clinical_trial": " in a clinical trial", "rct": " in a randomized controlled trial", "men_women": " in men and women",
      "mice": " in mice", "rats": " in rats", "animals": " in animals", "animal_models": " in animal models",
      "rodents": " in rodents", "vitro": " in vitro", "cell_lines": " in cell lines",
      "general": " in general", "detail": " in detail", "literature": " in the literature"}
S2 = {"children": " in children", "older_adults": " in older adults", "women": " in women", "men": " in men",
      "pregnancy": " in pregnancy", "rct_only": " randomized controlled trial", "case_report": " case report", "review": " review"}
SUFFIXES = {"base": "", **S1, **S2}
MODELS = os.environ.get("MODELS", "bm25,e5,qwen3-emb,medcpt,medcpt+ce,harrier,egemma2").split(",")


def rerank_ce(qt, cand_lists, full):
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(f"{WORKSPACE_ROOT}/models/medcpt-cross")
    ce = AutoModelForSequenceClassification.from_pretrained(f"{WORKSPACE_ROOT}/models/medcpt-cross", torch_dtype=torch.float16).cuda().eval()
    out = []
    pairs = [(i, j) for i, c in enumerate(cand_lists) for j in c]
    scores = np.zeros(len(pairs), np.float32)
    order = sorted(range(len(pairs)), key=lambda k: len(full[pairs[k][1]]))
    for s in range(0, len(order), 256):
        idx = order[s:s + 256]
        e = tok([[qt[pairs[k][0]], full[pairs[k][1]]] for k in idx], truncation=True, padding=True, max_length=512,
                return_tensors="pt").to("cuda")
        with torch.inference_mode():
            scores[idx] = ce(**e).logits.squeeze(-1).float().cpu().numpy()
    pos = 0
    for i, c in enumerate(cand_lists):
        sc = scores[pos:pos + len(c)]; pos += len(c)
        o = np.argsort(-sc)
        out.append([(c[k], float(sc[k])) for k in o])
    del ce
    torch.cuda.empty_cache()
    return out


for ds in (sys.argv[1:] if __name__ == "__main__" else []):  # importable for its suffix lists
    ids, titles, texts, qs, qrels = load(ds)
    qids = sorted(qs)
    full = [f"{t} {x}".strip() for t, x in zip(titles, texts)]
    runs = []
    names = list(SUFFIXES)
    qtexts = {n: [qs[q] + SUFFIXES[n] for q in qids] for n in names}
    if "bm25" in MODELS:
        import bm25s
        retr = bm25s.BM25()
        retr.index(bm25s.tokenize(full, stopwords="en", show_progress=False), show_progress=False)
        for n in names:
            res, sc = retr.retrieve(bm25s.tokenize(qtexts[n], stopwords="en", show_progress=False), k=K, show_progress=False, n_threads=4)
            for i, q in enumerate(qids):
                runs += [(ds, "bm25", n, q, ids[j], r, float(s)) for r, (j, s) in enumerate(zip(res[i], sc[i]))]
        del retr
        print(ds, "bm25 done", flush=True)
    for m in [x for x in MODELS if x in ("e5", "qwen3-emb", "medcpt", "harrier", "egemma2")]:
        docs = list(zip(titles, texts)) if m in ("medcpt", "egemma2") else full
        De = encode(m, docs, "doc", cache_key=f"p04_{ds}")
        allq = [x for n in names for x in qtexts[n]]
        Qe = encode(m, allq, "query", bs=64)
        I, S = search(Qe, De, k=K)
        for b, n in enumerate(names):
            for i, q in enumerate(qids):
                row = b * len(qids) + i
                runs += [(ds, m, n, q, ids[j], r, float(s)) for r, (j, s) in enumerate(zip(I[row], S[row]))]
            if m == "medcpt" and "medcpt+ce" in MODELS and n in S1 or (m == "medcpt" and "medcpt+ce" in MODELS and n == "base"):
                rr = rerank_ce(qtexts[n], [list(I[b * len(qids) + i]) for i in range(len(qids))], full)
                for i, q in enumerate(qids):
                    runs += [(ds, "medcpt+ce", n, q, ids[j], r, s) for r, (j, s) in enumerate(rr[i])]
        free()
        print(ds, m, "done", flush=True)
    pd.DataFrame(runs, columns=["ds", "retriever", "scope", "qid", "doc", "rank", "score"]).to_parquet(f"{ROOT}/results/runs_sweep_{ds}.parquet")
    print(ds, "written", len(runs), flush=True)
