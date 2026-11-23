"""P04 S9: more retriever families, same queries and suffixes as sweep.py.

  splade     learned sparse: SPLADE++ (prithivida/Splade_PP_en_v1), sentence-transformers SparseEncoder,
             exact sparse dot products
  colbert    multi-vector: GTE-ModernColBERT-v1 (PyLate, run with envs/pylate), exact MaxSim over all documents
  qwen4b     a larger embedder: Qwen3-Embedding-4B (same instruction format as the 0.6B model)
  fusion     reciprocal-rank fusion (k = 60) of BM25 and Qwen3-Embedding from runs_sweep_<ds>.parquet
python families.py <kind> <ds> [...]   Output: results/runs_fam_<kind>_<ds>.parquet (sweep.py columns)
"""

from pathlib import Path

WORKSPACE_ROOT = Path(__file__).resolve().parents[1].as_posix()
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, f"{WORKSPACE_ROOT}/src")
from run_retrieval import load  # noqa: E402
from sweep import SUFFIXES  # noqa: E402

ROOT = WORKSPACE_ROOT
K = 100
M = f"{WORKSPACE_ROOT}/models"
QTASK = "Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery: "


def topk_dense(Q, D, k=K, bs=256):
    import torch
    Dt = torch.from_numpy(np.ascontiguousarray(D)).cuda().half()
    I, S = [], []
    for s in range(0, len(Q), bs):
        sc = torch.from_numpy(np.ascontiguousarray(Q[s:s + bs])).cuda().half() @ Dt.T
        v, i = sc.float().topk(min(k, D.shape[0]), dim=1)
        I.append(i.cpu().numpy()); S.append(v.cpu().numpy())
    return np.concatenate(I), np.concatenate(S)


def emit(runs, ds, name, names, qids, ids, I, S):
    for b, n in enumerate(names):
        for i, q in enumerate(qids):
            row = b * len(qids) + i
            runs += [(ds, name, n, q, ids[j], r, float(s)) for r, (j, s) in enumerate(zip(I[row], S[row]))]


def run(kind, ds):
    ids, titles, texts, qs, qrels = load(ds)
    qids = sorted(qs)
    full = [f"{t} {x}".strip() for t, x in zip(titles, texts)]
    names = list(SUFFIXES)
    allq = [qs[q] + SUFFIXES[n] for n in names for q in qids]
    runs = []
    if kind == "qwen4b":
        import torch
        from sentence_transformers import SentenceTransformer
        m = SentenceTransformer(f"{M}/qwen3-emb-4b", device="cuda", model_kwargs={"torch_dtype": torch.bfloat16})
        m.max_seq_length = 512
        cp = f"{WORKSPACE_ROOT}/shared/emb_cache/qwen3-emb-4b/p04_{ds}_doc.npy"
        if os.path.exists(cp):
            D = np.load(cp)
        else:
            order = np.argsort([len(x) for x in full])
            D = np.zeros((len(full), m.get_sentence_embedding_dimension()), np.float16)
            D[order] = m.encode([full[i] for i in order], batch_size=32, normalize_embeddings=True, show_progress_bar=False).astype(np.float16)
            os.makedirs(os.path.dirname(cp), exist_ok=True); np.save(cp, D)
        Q = m.encode([QTASK + x for x in allq], batch_size=64, normalize_embeddings=True, show_progress_bar=False).astype(np.float16)
        I, S = topk_dense(Q, D)
        emit(runs, ds, "qwen3-emb-4b", names, qids, ids, I, S)
    elif kind == "splade":
        import scipy.sparse as sp
        import torch
        from sentence_transformers import SparseEncoder
        m = SparseEncoder(f"{M}/splade-pp", device="cuda")

        def enc(xs, query):
            f = m.encode_query if query else m.encode_document
            rows = []
            for s in range(0, len(xs), 512):
                e = f(xs[s:s + 512], batch_size=64, convert_to_sparse_tensor=True).coalesce().cpu()
                i, v = e.indices().numpy(), e.values().float().numpy()
                rows.append(sp.csr_matrix((v, (i[0], i[1])), shape=(e.shape[0], e.shape[1])))
            return sp.vstack(rows).tocsr()
        Dm = enc(full, False)
        Qm = enc(allq, True)
        Sc = (Qm @ Dm.T).tocsr()
        I = np.zeros((Qm.shape[0], K), int); S = np.zeros((Qm.shape[0], K), np.float32)
        for r in range(Qm.shape[0]):
            row = Sc.getrow(r); cols, vals = row.indices, row.data
            o = np.argsort(-vals)[:K]
            I[r, :len(o)] = cols[o]; S[r, :len(o)] = vals[o]
        emit(runs, ds, "splade-pp", names, qids, ids, I, S)
    elif kind == "colbert":
        import torch
        from pylate import models
        m = models.ColBERT(model_name_or_path=f"{M}/gte-moderncolbert", device="cuda")
        docs = m.encode(full, is_query=False, batch_size=64, show_progress_bar=False)
        L = max(len(d) for d in docs)
        Dp = torch.zeros((len(docs), L, docs[0].shape[1]), dtype=torch.float16)
        for i, d in enumerate(docs):
            Dp[i, :len(d)] = torch.as_tensor(d, dtype=torch.float16)
        Dp = Dp.cuda()
        Q = m.encode(allq, is_query=True, batch_size=128, show_progress_bar=False)
        I = np.zeros((len(Q), K), int); S = np.zeros((len(Q), K), np.float32)
        qb = 64
        for s in range(0, len(Q), qb):
            qt = torch.nn.utils.rnn.pad_sequence([torch.as_tensor(x, dtype=torch.float16) for x in Q[s:s + qb]],
                                                 batch_first=True).cuda()  # (b, lq, d); zero rows score 0
            best = []
            for c in range(0, Dp.shape[0], 4000):
                sim = torch.einsum("bqd,nld->bnql", qt, Dp[c:c + 4000])  # padding rows are zero -> score 0
                best.append(sim.max(-1).values.float().sum(-1))
            sc = torch.cat(best, 1)
            v, i = sc.topk(min(K, sc.shape[1]), dim=1)
            I[s:s + len(qt)] = i.cpu().numpy(); S[s:s + len(qt)] = v.cpu().numpy()
        emit(runs, ds, "gte-moderncolbert", names, qids, ids, I, S)
    elif kind == "fusion":
        R = pd.read_parquet(f"{ROOT}/results/runs_sweep_{ds}.parquet")
        R = R[R.retriever.isin(["bm25", "qwen3-emb"])]
        R["rrf"] = 1.0 / (60 + R["rank"] + 1)
        F = R.groupby(["scope", "qid", "doc"]).rrf.sum().reset_index()
        for (sc, q), g in F.groupby(["scope", "qid"]):
            g = g.sort_values("rrf", ascending=False).head(K)
            runs += [(ds, "rrf(bm25,qwen3-emb)", sc, q, d, r, float(s)) for r, (d, s) in enumerate(zip(g.doc, g.rrf))]
    return runs


if __name__ == "__main__":
    kind = sys.argv[1]
    for ds in sys.argv[2:]:
        runs = run(kind, ds)
        pd.DataFrame(runs, columns=["ds", "retriever", "scope", "qid", "doc", "rank", "score"]).to_parquet(
            f"{ROOT}/results/runs_fam_{kind}_{ds}.parquet")
        print(ds, kind, "written", len(runs), flush=True)
