"""P04 step 3: species scope of retrieved evidence.

Document scope from MeSH: human_only (Humans, no Animals), animal_only (Animals, no Humans),
both, unknown (no MeSH / neither); in_vitro flag (cell lines, cultured cells, in vitro techniques).
Human-health queries: Qwen3.5-9B classifier, checked against gpt-4o-mini on 200 queries.
RQ1: top-10 composition for human-health queries vs the judged-relevant pool; within judged-
relevant docs, does being animal-only lower the score (query fixed effects, controls: BM25
score, log length)?  RQ2: scope sensitivity of appended "in humans / in mice / in vitro".
Fix: MeSH re-ranking (animal-only docs moved below the rest in the top-100) vs a text-only
scope matcher (regex on the abstract); Recall@10 and nDCG@10 before/after.
Output: results/*.csv
"""

from pathlib import Path

WORKSPACE_ROOT = Path(__file__).resolve().parents[1].as_posix()
import glob
import json
import os
import re
import sys

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

sys.path.insert(0, f"{WORKSPACE_ROOT}/shared")
from llm import LLM, parse_json  # noqa: E402
from metrics import ndcg_at_k  # noqa: E402

ROOT = WORKSPACE_ROOT
DSS = [d for d in ["nfcorpus", "bioasq", "trec-covid"]]
VITRO = re.compile(r"Cell Line|Cells, Cultured|In Vitro Techniques|Cell Culture", re.I)
ANIMAL_TXT = re.compile(r"\b(mice|mouse|murine|rats?|rodents?|rabbits?|dogs?|pigs?|primates?|monkeys?|zebrafish|drosophila|in vitro|cell lines?|cultured cells)\b", re.I)
HUMAN_TXT = re.compile(r"\b(patients?|participants?|subjects?|volunteers?|women|men|adults?|children|humans?|people|cohort|trial)\b", re.I)

mesh = {}
for l in open(f"{ROOT}/data/mesh.jsonl"):
    r = json.loads(l); mesh[r["pmid"]] = r


def scope_of(pmid):
    r = mesh.get(pmid)
    if not r or not r["mesh"]:
        return "unknown", False
    terms = [m.split("/")[0].strip("* ") for m in r["mesh"]]
    h, a = "Humans" in terms, "Animals" in terms
    v = any(VITRO.search(t) for t in terms)
    s = "both" if h and a else "human_only" if h else "animal_only" if a else ("vitro_only" if v else "unknown")
    return s, v


rows_cls = []
llm = LLM("local", cache_name="p04_cls")
api = LLM("gpt-4o-mini", cache_name="p04_cls_api", concurrency=16)
luna = LLM("gpt-6-luna", cache_name="p04_cls_luna", concurrency=16)
CLS = ("Is the following search query about human health, i.e. does the searcher most likely want evidence about "
       "effects in people (not animals, cells or the environment)? Query: \"{q}\"\nReply with JSON {{\"human\": true/false}}.")
all_runs, all_rel = [], []
for ds in DSS:
    try:
        R = pd.read_parquet(f"{ROOT}/results/runs_{ds}.parquet")
        if os.path.exists(f"{ROOT}/results/runs_new_{ds}.parquet"):  # newest embedders
            R = pd.concat([R, pd.read_parquet(f"{ROOT}/results/runs_new_{ds}.parquet")], ignore_index=True)
        for f_ in sorted(glob.glob(f"{ROOT}/results/runs_extra_{ds}_*.parquet")):  # "in patients" / "in human participants"
            R = pd.concat([R, pd.read_parquet(f_)], ignore_index=True)
    except FileNotFoundError:
        continue
    d2p = json.load(open(f"{ROOT}/data/doc2pmid_{ds}.json"))
    qq = json.load(open(f"{ROOT}/data/qq_{ds}.json"))
    qids = sorted(qq["queries"])
    outs = llm.chat_many([[{"role": "user", "content": CLS.format(q=qq["queries"][q])}] for q in qids], max_tokens=20)
    hum = {}
    for q, o in zip(qids, outs):
        j = parse_json(o[0]["text"]) or {}
        hum[q] = bool(j.get("human", False)) and not ANIMAL_TXT.search(qq["queries"][q])
    rng = np.random.default_rng(0)
    samp = list(rng.choice(qids, min(200, len(qids)), replace=False))
    ja = api.chat_many([[{"role": "user", "content": CLS.format(q=qq["queries"][q])}] for q in samp], max_tokens=20)
    agree = np.mean([hum[q] == bool((parse_json(o[0]["text"]) or {}).get("human", False)) for q, o in zip(samp, ja)])
    # second judge: gpt-6-luna (reasoning model, needs room for hidden tokens)
    jl = luna.chat_many([[{"role": "user", "content": CLS.format(q=qq["queries"][q])}] for q in samp], max_tokens=600)
    agree_l = np.mean([hum[q] == bool((parse_json(o[0]["text"]) or {}).get("human", False)) for q, o in zip(samp, jl)])
    rows_cls.append({"ds": ds, "n_queries": len(qids), "human_share": np.mean(list(hum.values())),
                     "agreement_gpt4o_mini": agree, "agreement_gpt6_luna": agree_l})
    R["pmid"] = R.doc.map(d2p)
    sv = R.pmid.map(lambda p: scope_of(p) if isinstance(p, str) else ("nopmid", False))
    R["scope_cat"] = sv.map(lambda x: x[0])
    R["vitro"] = sv.map(lambda x: bool(x[1]))  # in-vitro flag, independent of the Humans/Animals tags
    R["human_q"] = R.qid.map(hum)
    all_runs.append(R)
    rel = pd.read_parquet(f"{ROOT}/results/relscores_{ds}.parquet")
    if os.path.exists(f"{ROOT}/results/relscores_new_{ds}.parquet") and not rel.empty:
        rel = pd.concat([rel, pd.read_parquet(f"{ROOT}/results/relscores_new_{ds}.parquet")], ignore_index=True)
    if rel.empty:
        print("no judged-relevant scores for", ds)
        continue
    rel["pmid"] = rel.doc.map(d2p)
    rel["scope_cat"] = rel.pmid.map(lambda p: scope_of(p)[0] if isinstance(p, str) else "nopmid")
    rel["human_q"] = rel.qid.map(hum)
    bm = rel[rel.retriever == "e5"][["qid", "doc", "bm25"]]
    rel = rel.drop(columns=["bm25"]).merge(bm, on=["qid", "doc"], how="left")
    all_rel.append(rel)
    # judged-relevant pool composition
pd.DataFrame(rows_cls).to_csv(f"{ROOT}/results/query_classification.csv", index=False)
print(pd.DataFrame(rows_cls))
R = pd.concat(all_runs)
REL = pd.concat(all_rel)
known = ["human_only", "animal_only", "both", "vitro_only"]

# RQ1a: top-10 composition, human-health queries, base scope
t10 = R[(R["rank"] < 10) & (R.scope == "base") & (R.human_q)]
comp = t10[t10.scope_cat.isin(known)].groupby(["ds", "retriever"]).scope_cat.value_counts(normalize=True).unstack().fillna(0)
pool = REL[(REL.retriever == "e5") & (REL.human_q) & (REL.scope_cat.isin(known))].groupby("ds").scope_cat.value_counts(normalize=True).unstack().fillna(0)
pool["retriever"] = "judged-relevant pool"
comp = pd.concat([comp.reset_index(), pool.reset_index()])
comp.to_csv(f"{ROOT}/results/top10_composition.csv", index=False)
print(comp.round(3).to_string(index=False))
cov = R[(R["rank"] < 10) & (R.scope == "base")].groupby("ds").scope_cat.apply(lambda s: s.isin(known).mean())
print("MeSH coverage in top-10:", cov.round(3).to_dict())

# RQ1c: per-query exposure (what a health-RAG user would meet): share of human-health queries whose
# top 10 holds >=1 / >=3 animal-only studies, and how many of those animal-only hits are judged relevant
qr_all = {}
for ds in DSS:
    try:
        qr_all[ds] = json.load(open(f"{ROOT}/data/qq_{ds}.json"))["qrels"]
    except FileNotFoundError:
        pass
B = 1000
rng = np.random.default_rng(0)


def boot_mean(x):
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    if len(x) == 0:
        return np.nan, np.nan, np.nan
    bs = rng.choice(x, (B, len(x))).mean(1)
    return float(x.mean()), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))


# per-query judged-relevant pool share of animal-only docs (human-health queries), for the excess test
pool_q = (REL[(REL.retriever == "e5") & (REL.human_q) & (REL.scope_cat.isin(known))]
          .assign(a=lambda d: (d.scope_cat == "animal_only").astype(float)).groupby(["ds", "qid"]).a.mean())
exp_rows = []
for (ds, m), g in t10.groupby(["ds", "retriever"]):
    per_q = g.groupby("qid").scope_cat.apply(lambda s: int((s == "animal_only").sum()))
    an = g[g.scope_cat == "animal_only"]
    rel_an = np.mean([qr_all.get(ds, {}).get(r.qid, {}).get(r.doc, 0) > 0 for r in an.itertuples()]) if len(an) else np.nan
    e1, e1lo, e1hi = boot_mean((per_q >= 1).astype(float))
    # excess: top-10 animal-only share minus the same query's judged-relevant animal-only share
    t_a = g[g.scope_cat.isin(known)].assign(a=lambda d: (d.scope_cat == "animal_only").astype(float)).groupby("qid").a.mean()
    pq_ = pool_q.loc[ds] if ds in pool_q.index.get_level_values(0) else pd.Series(dtype=float)
    common = t_a.index.intersection(pq_.index)
    ex, exlo, exhi = boot_mean((t_a.loc[common] - pq_.loc[common]).to_numpy())
    exp_rows.append({"ds": ds, "retriever": m, "n_queries": len(per_q), "q_with_1plus_animal_only": e1,
                     "q1_lo": e1lo, "q1_hi": e1hi, "q_with_3plus_animal_only": float((per_q >= 3).mean()),
                     "animal_only_hits_judged_relevant": float(rel_an), "n_animal_hits": len(an),
                     "animal_excess_vs_relevant_pool": ex, "excess_lo": exlo, "excess_hi": exhi, "n_q_excess": len(common)})
EX = pd.DataFrame(exp_rows)
EX.to_csv(f"{ROOT}/results/exposure.csv", index=False)
print(EX.round(3).to_string(index=False))

# RQ1b: within judged-relevant docs, animal-only vs human-only score gap (query FE, relevance grade FE).
# Only queries with >= 2 such docs carry within-query information (a single doc gives an undefined z).
REL["grade"] = [qr_all.get(d, {}).get(q, {}).get(doc, 1) for d, q, doc in zip(REL.ds, REL.qid, REL.doc)]
fits = []
for (ds, m), g in REL[(REL.human_q) & (REL.scope_cat.isin(["human_only", "animal_only"]))].groupby(["ds", "retriever"]):
    g = g[g.groupby("qid").doc.transform("count") >= 2].copy()
    if g.scope_cat.nunique() < 2 or len(g) < 30:
        continue
    g["animal"] = (g.scope_cat == "animal_only").astype(int)
    g["z"] = g.groupby("qid").score.transform(lambda s: (s - s.mean()) / (s.std() + 1e-9))
    g["bmz"] = g.groupby("qid").bm25.transform(lambda s: (s - s.mean()) / (s.std() + 1e-9)).fillna(0)
    g["ll"] = np.log1p(g["len"])
    g = g.dropna(subset=["z", "bmz", "ll"])
    n_q_both = int((g.groupby("qid").animal.nunique() == 2).sum())  # queries that identify the contrast
    form = "z ~ animal + bmz + ll + C(qid)" + (" + C(grade)" if g.grade.nunique() > 1 else "")
    try:
        f = smf.ols(form, g).fit(cov_type="cluster", cov_kwds={"groups": pd.factorize(g.qid)[0]})
        fits.append({"ds": ds, "retriever": m, "n": len(g), "n_animal": int(g.animal.sum()), "n_q_both": n_q_both,
                     "grade_fe": g.grade.nunique() > 1, "coef_animal": f.params["animal"],
                     "lo": f.conf_int().loc["animal", 0], "hi": f.conf_int().loc["animal", 1], "p": f.pvalues["animal"]})
    except Exception as e:  # noqa: BLE001
        print("fit fail", ds, m, e)
F = pd.DataFrame(fits)
F.to_csv(f"{ROOT}/results/within_relevant_gap.csv", index=False)
print(F.round(3).to_string(index=False))

# RQ2: scope sensitivity. Per query: share of human-only / animal-only / in-vitro-flagged docs among the
# MeSH-labelled top 10, variant minus base; mean over queries with a query-bootstrap 95% CI.
sens = []
SHIFTS = {"human_shift_humans": ("h", "humans"), "animal_shift_humans": ("a", "humans"),
          "animal_shift_mice": ("a", "mice"), "human_shift_mice": ("h", "mice"), "vitro_shift_vitro": ("v", "vitro"),
          "human_shift_patients": ("h", "patients"), "animal_shift_patients": ("a", "patients"),
          "human_shift_participants": ("h", "participants"), "animal_shift_participants": ("a", "participants")}
for (ds, m), g in R[(R["rank"] < 10) & (R.scope_cat.isin(known))].groupby(["ds", "retriever"]):
    g = g.assign(h=(g.scope_cat == "human_only").astype(float), a=(g.scope_cat == "animal_only").astype(float),
                 v=g.vitro.astype(float))
    P_ = g.groupby(["qid", "scope"])[["h", "a", "v"]].mean().unstack("scope")
    row = {"ds": ds, "retriever": m, "n_q": len(P_)}
    for name, (met, var) in SHIFTS.items():
        if (met, var) in P_.columns and (met, "base") in P_.columns:
            row[name], row[name + "_lo"], row[name + "_hi"] = boot_mean((P_[(met, var)] - P_[(met, "base")]).to_numpy())
        else:
            row[name] = row[name + "_lo"] = row[name + "_hi"] = np.nan
    for met, col in (("a", "animal_base"), ("h", "human_base"), ("v", "vitro_base")):
        row[col] = float(P_[(met, "base")].mean()) if (met, "base") in P_.columns else np.nan
    sens.append(row)
S = pd.DataFrame(sens)
S.to_csv(f"{ROOT}/results/scope_sensitivity.csv", index=False)
print(S.round(3).to_string(index=False))

# Fix: MeSH demotion vs text demotion of animal-only docs, human-health queries
qrels = {}
for ds in DSS:
    try:
        qq = json.load(open(f"{ROOT}/data/qq_{ds}.json"))
    except FileNotFoundError:
        continue
    for q, d in qq["qrels"].items():
        qrels[(ds, q)] = d
texts = {}
need = set(R[(R.scope == "base") & (R.human_q)].doc)
p = pd.read_parquet(f"{ROOT}/data/rag-mini-bioasq/data/passages.parquet/part.0.parquet")
texts.update({str(i): t for i, t in zip(p.index, p.passage) if str(i) in need})
for ds in ["nfcorpus", "trec-covid"]:
    c = pd.read_parquet(f"{WORKSPACE_ROOT}/data/beir/{ds}/corpus")
    c = c[c["_id"].astype(str).isin(need)]
    texts.update({str(i): f"{a} {b}" for i, a, b in zip(c["_id"], c["title"].fillna(""), c["text"].fillna(""))})


def text_animal(d):
    t = texts.get(d, "")
    return bool(ANIMAL_TXT.search(t)) and not HUMAN_TXT.search(t)


fix_rows = []
for (ds, m, q), g in R[(R.scope == "base") & (R.human_q)].groupby(["ds", "retriever", "qid"]):
    g = g.sort_values("rank")
    docs = g.doc.tolist(); cats = g.scope_cat.tolist()
    rel = qrels.get((ds, q), {})
    mesh_order = [d for d, c in zip(docs, cats) if c != "animal_only"] + [d for d, c in zip(docs, cats) if c == "animal_only"]
    ta = [text_animal(d) for d in docs]
    text_order = [d for d, a in zip(docs, ta) if not a] + [d for d, a in zip(docs, ta) if a]
    for name, order in (("none", docs), ("mesh_demote", mesh_order), ("text_demote", text_order)):
        top = order[:10]
        cs = dict(zip(docs, cats))
        fix_rows.append({"ds": ds, "retriever": m, "qid": q, "fix": name, "ndcg10": ndcg_at_k(order, rel, 10),
                         "rec10": len({d for d in top if rel.get(d, 0) > 0}) / max(1, sum(1 for v in rel.values() if v > 0)),
                         "animal10": np.mean([cs[d] == "animal_only" for d in top])})
FX = pd.DataFrame(fix_rows).groupby(["ds", "retriever", "fix"])[["ndcg10", "rec10", "animal10"]].mean().reset_index()
FX.to_csv(f"{ROOT}/results/fix_mesh.csv", index=False)
print(FX.round(4).to_string(index=False))
