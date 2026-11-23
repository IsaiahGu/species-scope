"""P04 step 2: top-100 rankings for base queries and scope-appended queries.

Datasets: BioASQ (mini corpus, 40k PubMed abstracts), NFCorpus (test), TREC-COVID.
Retrievers: BM25, e5-base-v2, Qwen3-Embedding-0.6B, MedCPT (biomedical bi-encoder),
MedCPT cross-encoder reranking the MedCPT top-100.
Scope variants: base, "<q> in humans", "<q> in mice", "<q> in vitro".
Also scores of every judged-relevant document (for the within-relevant rank analysis).
Output: results/runs_<ds>.parquet, results/relscores_<ds>.parquet
"""

from pathlib import Path

WORKSPACE_ROOT = Path(__file__).resolve().parents[1].as_posix()
import json
import sys

import bm25s
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, f"{WORKSPACE_ROOT}/shared")
from embed import encode, search  # noqa: E402

ROOT = WORKSPACE_ROOT
B = f"{WORKSPACE_ROOT}/data/beir"
SCOPES = {"base": "", "humans": " in humans", "mice": " in mice", "vitro": " in vitro"}
K = 100


def load(ds):
    if ds == "bioasq":
        p = pd.read_parquet(f"{ROOT}/data/rag-mini-bioasq/data/passages.parquet/part.0.parquet")
        ids = [str(i) for i in p.index]; titles = [""] * len(ids); texts = p.passage.tolist()
        t = pd.read_parquet(f"{ROOT}/data/rag-mini-bioasq/data/test.parquet/part.0.parquet")
        qs = {str(i): q for i, q in zip(t.index, t.question)}
        import random as _r
        keep = set(_r.Random(0).sample(sorted(qs), min(1000, len(qs))))  # 1000-question sample keeps the cross-encoder cost bounded
        qs = {k: v for k, v in qs.items() if k in keep}
        qrels = {str(i): {str(x): 1 for x in (json.loads(rel) if isinstance(rel, str) else rel)} for i, rel in zip(t.index, t.relevant_passage_ids)}
    else:
        c = pd.read_parquet(f"{B}/{ds}/corpus")
        ids = c["_id"].astype(str).tolist(); titles = c["title"].fillna("").tolist(); texts = c["text"].fillna("").tolist()
        qd = pd.read_parquet(f"{B}/{ds}/queries").set_index("_id")["text"].to_dict()
        qr = pd.read_csv(f"{B}/{ds}/qrels/test.tsv", sep="\t", dtype=str)
        qrels = {}
        for r in qr.itertuples():
            qrels.setdefault(r[1], {})[r[2]] = int(r[3])
        qs = {k: qd[k] for k in qrels if k in qd}
    return ids, titles, texts, qs, qrels


def main(ds):
    ids, titles, texts, qs, qrels = load(ds)
    qids = sorted(qs)
    pos = {d: i for i, d in enumerate(ids)}
    full = [f"{t} {x}".strip() for t, x in zip(titles, texts)]
    runs, rel = [], []
    retr = bm25s.BM25()
    retr.index(bm25s.tokenize(full, stopwords="en", show_progress=False), show_progress=False)
    from embed import free
    enc, qenc = {}, {}
    for m in ["e5", "qwen3-emb", "medcpt"]:  # one model in GPU memory at a time
        enc[m] = encode(m, full if m != "medcpt" else list(zip(titles, texts)), "doc", cache_key=f"p04_{ds}")
        for scope, suf in SCOPES.items():
            qenc[(m, scope)] = encode(m, [qs[q] + suf for q in qids], "query")
        free()
    for scope, suf in SCOPES.items():
        qt = [qs[q] + suf for q in qids]
        res, sc = retr.retrieve(bm25s.tokenize(qt, stopwords="en", show_progress=False), k=K, show_progress=False, n_threads=32)
        for i, q in enumerate(qids):
            for r, (j, s) in enumerate(zip(res[i], sc[i])):
                runs.append((ds, "bm25", scope, q, ids[j], r, float(s)))
        for m, De in enc.items():
            Qe = qenc[(m, scope)]
            I, S = search(Qe, De, k=K)
            for i, q in enumerate(qids):
                for r, (j, s) in enumerate(zip(I[i], S[i])):
                    runs.append((ds, m, scope, q, ids[j], r, float(s)))
            if scope == "base":  # scores of all judged-relevant docs
                for i, q in enumerate(qids):
                    js = [pos[d] for d, g in qrels[q].items() if g > 0 and d in pos]
                    if js:
                        s = De[js].astype(np.float32) @ Qe[i].astype(np.float32)
                        bm = None
                        if m == "e5":
                            try:
                                toks = bm25s.tokenize([qt[i]], stopwords="en", show_progress=False, return_ids=False)[0]
                                toks = [t for t in toks if t in retr.vocab_dict]
                                bm = retr.get_scores(toks)[js] if toks else np.zeros(len(js))
                            except Exception:  # noqa: BLE001
                                bm = None
                        for k_, j in enumerate(js):
                            rel.append({"ds": ds, "retriever": m, "qid": q, "doc": ids[j], "grade": qrels[q][ids[j]],
                                        "score": float(s[k_]), "len": len(full[j].split()),
                                        "bm25": float(bm[k_]) if bm is not None else np.nan})
            if m == "medcpt":  # cross-encoder rerank of the MedCPT top-100
                from transformers import AutoModelForSequenceClassification, AutoTokenizer
                tok = AutoTokenizer.from_pretrained(f"{WORKSPACE_ROOT}/models/medcpt-cross")
                ce = AutoModelForSequenceClassification.from_pretrained(f"{WORKSPACE_ROOT}/models/medcpt-cross", torch_dtype=torch.float16).cuda().eval()
                for i, q in enumerate(qids):
                    cand = I[i]
                    pairs = [[qt[i], full[j]] for j in cand]
                    scs = []
                    for s0 in range(0, len(pairs), 50):
                        e = tok(pairs[s0:s0 + 50], truncation=True, padding=True, max_length=512, return_tensors="pt").to("cuda")
                        with torch.inference_mode():
                            scs += ce(**e).logits.squeeze(-1).float().cpu().tolist()
                    o = np.argsort(-np.array(scs))
                    for r, k_ in enumerate(o):
                        runs.append((ds, "medcpt+ce", scope, q, ids[cand[k_]], r, float(scs[k_])))
                del ce
                torch.cuda.empty_cache()
        print(ds, scope, "done", flush=True)
    pd.DataFrame(runs, columns=["ds", "retriever", "scope", "qid", "doc", "rank", "score"]).to_parquet(f"{ROOT}/results/runs_{ds}.parquet")
    pd.DataFrame(rel).to_parquet(f"{ROOT}/results/relscores_{ds}.parquet")
    json.dump({"queries": qs, "qrels": qrels}, open(f"{ROOT}/data/qq_{ds}.json", "w"))


if __name__ == "__main__":
    for ds in sys.argv[1:] or ["nfcorpus", "bioasq", "trec-covid"]:
        main(ds)
