"""P04 S8: does exposure to animal studies depend on corpus size?

Corpora: the BioASQ mini corpus (40,221 abstracts) alone, plus random PubMed baseline abstracts up to 200,000 and
2,000,000 documents (data/pubmed_sample.jsonl.gz; records with abstract and MeSH; BioASQ PMIDs removed). Queries:
the same 1,000 BioASQ questions; judged-relevant documents are all in the 40k core, added documents count as
non-relevant. Scopes: base, " in humans", " in patients", " in mice", " in vitro", " in general".
Retrievers: BM25 (bm25s, CPU) and two dense models (e5-base-v2, MedCPT; GPU).
Readout: exposure (human-health queries with >= 1 animal-only study in the top 10), top-10 animal-only and
human-only shares, scope shifts, nDCG@10.
python scale.py bm25 | dense <e5|medcpt>   Output: results/runs_scale_<retriever>.parquet
"""

from pathlib import Path

WORKSPACE_ROOT = Path(__file__).resolve().parents[1].as_posix()
import gzip
import json
import os
import random
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, f"{WORKSPACE_ROOT}/shared")
sys.path.insert(0, f"{WORKSPACE_ROOT}/src")
from run_retrieval import load  # noqa: E402

ROOT = WORKSPACE_ROOT
SC = {"base": "", "humans": " in humans", "patients": " in patients", "mice": " in mice", "vitro": " in vitro", "general": " in general"}
SIZES = [40221, 200000, 2000000]
K = 100


def corpus():
    ids, titles, texts, qs, qrels = load("bioasq")
    core = set(ids)
    extra = []
    for l in gzip.open(f"{ROOT}/data/pubmed_sample.jsonl.gz", "rt"):
        j = json.loads(l)
        if j["pmid"] not in core:
            extra.append(j)
    random.Random(0).shuffle(extra)
    extra = extra[:SIZES[-1] - len(ids)]
    with open(f"{ROOT}/data/scale_extra_mesh.jsonl", "w") as f:  # MeSH of the added documents
        for j in extra:
            f.write(json.dumps({"pmid": j["pmid"], "mesh": j["mesh"], "pt": j["pt"], "year": j["year"]}) + "\n")
    all_ids = list(ids) + [j["pmid"] for j in extra]
    all_txt = [f"{t} {x}".strip() for t, x in zip(titles, texts)] + [f"{j['title']} {j['abstract']}" for j in extra]
    pairs = list(zip(titles, texts)) + [(j["title"], j["abstract"]) for j in extra]
    return all_ids, all_txt, pairs, qs, len(ids)


def run_bm25():
    import bm25s
    ids, txt, _, qs, ncore = corpus()
    qids = sorted(qs)
    runs = []
    for n in SIZES:
        n = min(n, len(ids))
        retr = bm25s.BM25()
        retr.index(bm25s.tokenize(txt[:n], stopwords="en", show_progress=False), show_progress=False)
        for s, suf in SC.items():
            res, sc = retr.retrieve(bm25s.tokenize([qs[q] + suf for q in qids], stopwords="en", show_progress=False), k=K,
                                    show_progress=False, n_threads=4)
            for i, q in enumerate(qids):
                runs += [(n, "bm25", s, q, ids[j], r, float(v)) for r, (j, v) in enumerate(zip(res[i], sc[i]))]
        print("bm25 size", n, "done", flush=True)
        del retr
    pd.DataFrame(runs, columns=["size", "retriever", "scope", "qid", "doc", "rank", "score"]).to_parquet(f"{ROOT}/results/runs_scale_bm25.parquet")


def run_dense(m):
    from embed import encode, search
    ids, txt, pairs, qs, ncore = corpus()
    qids = sorted(qs)
    docs = pairs if m == "medcpt" else txt
    D = encode(m, docs, "doc", cache_key="p04_scale2m", bs=256)
    Q = encode(m, [qs[q] + suf for suf in SC.values() for q in qids], "query", bs=256)
    runs = []
    for n in SIZES:
        n = min(n, len(ids))
        I, S = search(Q, D[:n], k=K)
        for b, s in enumerate(SC):
            for i, q in enumerate(qids):
                row = b * len(qids) + i
                runs += [(n, m, s, q, ids[j], r, float(v)) for r, (j, v) in enumerate(zip(I[row], S[row]))]
        print(m, "size", n, "done", flush=True)
    pd.DataFrame(runs, columns=["size", "retriever", "scope", "qid", "doc", "rank", "score"]).to_parquet(f"{ROOT}/results/runs_scale_{m}.parquet")


if __name__ == "__main__":
    if sys.argv[1] == "bm25":
        run_bm25()
    else:
        run_dense(sys.argv[2])
