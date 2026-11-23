"""P04 S4 (population by instruction) and S8 (corpus size): MeSH composition of the top 10 for human-health queries.

Per query (human-health queries, classifier as in analyze.py, from its cache): share of human-only and of
animal-only studies among the MeSH-labelled top 10, exposure (>= 1 animal-only study in the top 10), nDCG@10.
S4: embedder task instructions (runs_instr_emb), Qwen3-Reranker over the top 100 (runs_instr_rerank), Qwen3.5-9B
    pointwise rerank of the top 20 (runs_instr_llm); each against its default instruction, next to the " in humans"
    and " in patients" suffixes of the same first-stage retriever (runs_sweep); paired over queries, 95% bootstrap CI.
S8: BioASQ questions over 40k / 200k / 2M documents (runs_scale_*; added PubMed documents carry their own MeSH and
    are never judged relevant); per size and retriever, the base readouts and the shift of each suffix vs base.
python s4_s8_analysis.py   Output: results/s4_instructions.csv, results/s8_scale.csv (stdout: results/s4_s8_output.txt)
"""

from pathlib import Path

WORKSPACE_ROOT = Path(__file__).resolve().parents[1].as_posix()
import json
import os
import re
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, f"{WORKSPACE_ROOT}/shared")
sys.path.insert(0, f"{WORKSPACE_ROOT}/src")
from llm import LLM, parse_json  # noqa: E402
from metrics import ndcg_at_k  # noqa: E402

ROOT = WORKSPACE_ROOT
rng = np.random.default_rng(0)
VITRO = re.compile(r"Cell Line|Cells, Cultured|In Vitro Techniques|Cell Culture", re.I)
ANIMAL_TXT = re.compile(r"\b(mice|mouse|murine|rats?|rodents?|rabbits?|dogs?|pigs?|primates?|monkeys?|zebrafish|drosophila|in vitro|cell lines?|cultured cells)\b", re.I)
CLS = ("Is the following search query about human health, i.e. does the searcher most likely want evidence about "
       "effects in people (not animals, cells or the environment)? Query: \"{q}\"\nReply with JSON {{\"human\": true/false}}.")
KNOWN = ("human_only", "animal_only", "both", "vitro_only")
mesh = {}
for l in open(f"{ROOT}/data/mesh.jsonl"):
    r = json.loads(l); mesh[r["pmid"]] = r


def cat(pmid):
    r = mesh.get(pmid)
    if not r or not r["mesh"]:
        return "unknown"
    t = {m.split("/")[0].strip("* ") for m in r["mesh"]}
    h, a = "Humans" in t, "Animals" in t
    v = any(VITRO.search(x) for x in t)
    return "both" if h and a else "human_only" if h else "animal_only" if a else ("vitro_only" if v else "unknown")


def human_flags(ds):
    qq = json.load(open(f"{ROOT}/data/qq_{ds}.json"))
    qids = sorted(qq["queries"])
    outs = LLM("local", cache_name="p04_cls").chat_many([[{"role": "user", "content": CLS.format(q=qq["queries"][q])}] for q in qids],
                                                         max_tokens=20, progress=False)
    hum = {q: bool((parse_json(o[0]["text"]) or {}).get("human", False)) and not ANIMAL_TXT.search(qq["queries"][q]) for q, o in zip(qids, outs)}
    return hum, qq["qrels"]


def per_query(R, d2p, qrels, keys):
    """R: rows of one run family; returns {key tuple: {qid: (human_only, animal_only, exposed, ndcg10)}}."""
    out = {}
    for k, g in R.groupby(keys):
        k = k if isinstance(k, tuple) else (k,)
        res = {}
        for q, gq in g.groupby("qid"):
            docs = gq.sort_values("rank").doc.tolist()
            c = [cat(d2p.get(d, d)) for d in docs[:10]]
            kn = [x for x in c if x in KNOWN]
            res[q] = (np.mean([x == "human_only" for x in kn]) if kn else np.nan, np.mean([x == "animal_only" for x in kn]) if kn else np.nan,
                      float(any(x == "animal_only" for x in c)), ndcg_at_k(docs, qrels.get(q, {}), 10))
        out[k] = res
    return out


def boot(x):
    x = np.asarray(x, float); x = x[~np.isnan(x)]
    if not len(x):
        return [np.nan, np.nan, np.nan]
    bs = x[rng.integers(0, len(x), (1000, len(x)))].mean(1)
    return [float(x.mean()), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]


MET = ("human_only", "animal_only", "exposed", "ndcg10")


def compare(rows, label, ds, cur, ref, extra):
    qs = [q for q in cur if q in ref]
    r = {"ds": ds, "comparison": label, "n_q": len(qs), **extra}
    for i, m in enumerate(MET):
        r[f"{m}_ref"] = float(np.nanmean([ref[q][i] for q in qs])) if qs else np.nan
        r[f"{m}_cur"] = float(np.nanmean([cur[q][i] for q in qs])) if qs else np.nan
        r[f"d_{m}"], r[f"d_{m}_lo"], r[f"d_{m}_hi"] = boot([cur[q][i] - ref[q][i] for q in qs])
    rows.append(r)


# ---- S4
S4 = []
for ds in ("nfcorpus", "bioasq", "trec-covid"):
    hum, qrels = human_flags(ds)
    d2p = json.load(open(f"{ROOT}/data/doc2pmid_{ds}.json"))
    H = {q for q, v in hum.items() if v}
    sw = pd.read_parquet(f"{ROOT}/results/runs_sweep_{ds}.parquet")
    sw = sw[sw.qid.isin(H) & sw.scope.isin(["base", "humans", "patients"]) & (sw["rank"] < 10)]
    SW = per_query(sw, d2p, qrels, ["retriever", "scope"])
    for kind in ("emb", "rerank", "llm"):
        fn = f"{ROOT}/results/runs_instr_{kind}_{ds}.parquet"
        if not os.path.exists(fn):
            print("missing", fn); continue
        X = pd.read_parquet(fn)
        X = X[X.qid.isin(H) & (X["rank"] < 10)]
        P = per_query(X, d2p, qrels, ["retriever", "scope"])
        for (ret, sc), cur in P.items():
            first = ret.split("@")[-1]
            default = "llm_relevant" if kind == "llm" else "instr_default"
            if sc != default and (ret, default) in P:
                compare(S4, f"{sc} vs {default}", ds, cur, P[(ret, default)], {"kind": kind, "retriever": ret})
            if sc == default and (first, "base") in SW:  # what the default reranking / instruction does by itself
                compare(S4, f"{sc} vs first-stage base", ds, cur, SW[(first, "base")], {"kind": kind, "retriever": ret})
            if sc == "instr_humans" and (first, "base") in SW:
                compare(S4, "instr_humans vs first-stage base", ds, cur, SW[(first, "base")], {"kind": kind, "retriever": ret})
        for ret in sorted({r for r, _ in P}):
            first = ret.split("@")[-1]
            for sfx in ("humans", "patients"):
                if (first, sfx) in SW and (first, "base") in SW:
                    compare(S4, f"suffix {sfx} vs base", ds, SW[(first, sfx)], SW[(first, "base")], {"kind": "suffix", "retriever": first})
S4 = pd.DataFrame(S4).drop_duplicates(["ds", "comparison", "kind", "retriever"])
S4.to_csv(f"{ROOT}/results/s4_instructions.csv", index=False)
pd.set_option("display.width", 250); pd.set_option("display.max_rows", 500)
cols = ["ds", "kind", "retriever", "comparison", "n_q", "animal_only_ref", "d_animal_only", "d_animal_only_lo", "d_animal_only_hi",
        "human_only_ref", "d_human_only", "d_exposed", "ndcg10_ref", "d_ndcg10", "d_ndcg10_lo", "d_ndcg10_hi"]
print("== S4 instructions (human-health queries; top 10; paired over queries, 95% CI)")
print(S4[cols].round(3).to_string(index=False))

# ---- S8
for l in open(f"{ROOT}/data/scale_extra_mesh.jsonl"):
    r = json.loads(l); mesh.setdefault(r["pmid"], r)
hum, qrels = human_flags("bioasq")
H = {q for q, v in hum.items() if v}
d2p = json.load(open(f"{ROOT}/data/doc2pmid_bioasq.json"))
S8 = []
for m in ("bm25", "e5", "medcpt"):
    fn = f"{ROOT}/results/runs_scale_{m}.parquet"
    if not os.path.exists(fn):
        print("missing", fn); continue
    X = pd.read_parquet(fn)
    X = X[X.qid.isin(H) & (X["rank"] < 10)]
    P = per_query(X, d2p, qrels, ["size", "scope"])
    for (size, sc), cur in P.items():
        r = {"retriever": m, "size": int(size), "scope": sc, "n_q": len(cur)}
        for i, k in enumerate(MET):
            r[k], r[f"{k}_lo"], r[f"{k}_hi"] = boot([v[i] for v in cur.values()])
        if sc != "base" and (size, "base") in P:
            b = P[(size, "base")]; qs = [q for q in cur if q in b]
            for i, k in enumerate(MET):
                r[f"d_{k}"], r[f"d_{k}_lo"], r[f"d_{k}_hi"] = boot([cur[q][i] - b[q][i] for q in qs])
        S8.append(r)
S8 = pd.DataFrame(S8).sort_values(["retriever", "scope", "size"])
S8.to_csv(f"{ROOT}/results/s8_scale.csv", index=False)
print("\n== S8 corpus size (BioASQ human-health questions; top 10; 95% CI)")
print(S8[["retriever", "scope", "size", "n_q", "exposed", "exposed_lo", "exposed_hi", "animal_only", "human_only", "ndcg10",
          "d_human_only", "d_human_only_lo", "d_human_only_hi", "d_animal_only"]].round(3).to_string(index=False))
