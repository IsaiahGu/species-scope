"""P04 S1/S2/S3 analysis of the suffix sweep.

S1 step 1  term log-odds: for ~40 population words, the share of human-only vs animal-only abstracts
           (MeSH) that contain the word; log odds ratio with a Woolf 95% CI (results/term_logodds.csv).
S1 step 3  per suffix and retriever: shift of the target share in the MeSH-labelled top 10 (human-only for
           human-pointing suffixes, animal-only for animal-pointing ones, the in-vitro flag for in-vitro ones)
           minus the mean shift of the three neutral suffixes, regressed on the log odds of the suffix's core
           word (one line per retriever: slope, R^2) (results/logodds_vs_shift.csv).
S2         other population axes: tag share (age, sex, pregnancy, publication type from MeSH) in the top 10,
           shift minus neutral, against the log odds of the core word in tagged vs untagged abstracts; the
           "default" side (adults, humans) vs the marked side (children, mice).
S3         retrieval cost of every suffix: nDCG@10, Recall@10, Recall@100 change vs the query alone, paired
           over queries (bootstrap 95% CI); human-health queries and all queries side by side.
python sweep_analyze.py   Output: results/term_logodds.csv, results/sweep_shifts.csv, results/logodds_vs_shift.csv,
                                  results/sweep_cost.csv, results/sweep_summary.json
"""

from pathlib import Path

WORKSPACE_ROOT = Path(__file__).resolve().parents[1].as_posix()
import glob
import json
import re
import sys

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

sys.path.insert(0, f"{WORKSPACE_ROOT}/shared")
sys.path.insert(0, f"{WORKSPACE_ROOT}/src")
from llm import LLM, parse_json  # noqa: E402
from metrics import ndcg_at_k  # noqa: E402
from run_retrieval import load  # noqa: E402
from sweep import S1, S2  # noqa: E402

ROOT = WORKSPACE_ROOT
DSS = ["bioasq", "nfcorpus", "trec-covid"]
rng = np.random.default_rng(0)
B = 1000
VITRO = re.compile(r"Cell Line|Cells, Cultured|In Vitro Techniques|Cell Culture", re.I)
ANIMAL_TXT = re.compile(r"\b(mice|mouse|murine|rats?|rodents?|rabbits?|dogs?|pigs?|primates?|monkeys?|zebrafish|drosophila|in vitro|cell lines?|cultured cells)\b", re.I)
TERMS = ["humans", "human", "people", "patients", "participants", "subjects", "volunteers", "adults", "men", "women",
         "children", "infants", "elderly", "older adults", "pregnant", "pregnancy", "trial", "clinical trial",
         "randomized controlled trial", "cohort", "case report", "review", "mice", "mouse", "rats", "rat", "rodents",
         "animals", "animal models", "murine", "in vitro", "cell line", "cell lines", "cultured", "dogs", "pigs",
         "rabbits", "zebrafish", "in general", "in detail", "the literature"]
CORE = {"humans": "humans", "people": "people", "patients": "patients", "adults": "adults", "participants": "participants",
        "subjects": "subjects", "volunteers": "volunteers", "clinical_trial": "clinical trial", "rct": "randomized controlled trial",
        "men_women": ("men", "women"), "mice": "mice", "rats": "rats", "animals": "animals", "animal_models": "animal models",
        "rodents": "rodents", "vitro": "in vitro", "cell_lines": "cell lines", "children": "children",
        "older_adults": "older adults", "women": "women", "men": "men", "pregnancy": "pregnancy",
        "rct_only": "randomized controlled trial", "case_report": "case report", "review": "review"}
HUMAN_SFX = ["humans", "people", "patients", "adults", "participants", "subjects", "volunteers", "clinical_trial", "rct", "men_women"]
ANIMAL_SFX = ["mice", "rats", "animals", "animal_models", "rodents"]
VITRO_SFX = ["vitro", "cell_lines"]
NEUTRAL = ["general", "detail", "literature"]
AXES = {"children": lambda m, p: bool({"Child", "Child, Preschool", "Infant", "Infant, Newborn", "Adolescent"} & m),
        "older_adults": lambda m, p: bool({"Aged", "Aged, 80 and over"} & m),
        "adults": lambda m, p: bool({"Adult", "Young Adult", "Middle Aged"} & m),
        "women": lambda m, p: "Female" in m and "Male" not in m, "men": lambda m, p: "Male" in m and "Female" not in m,
        "pregnancy": lambda m, p: "Pregnancy" in m, "rct_only": lambda m, p: "Randomized Controlled Trial" in p,
        "case_report": lambda m, p: "Case Reports" in p, "review": lambda m, p: bool({"Review", "Systematic Review"} & p)}

mesh = {}
for l in open(f"{ROOT}/data/mesh.jsonl"):
    r = json.loads(l); mesh[r["pmid"]] = r


def doc_info(pmid):
    r = mesh.get(pmid)
    if not r or not r["mesh"]:
        return None
    terms = {m.split("/")[0].strip("* ") for m in r["mesh"]}
    pt = set(r.get("pt") or [])
    h, a = "Humans" in terms, "Animals" in terms
    v = any(VITRO.search(t) for t in terms)
    cat = "both" if h and a else "human_only" if h else "animal_only" if a else ("vitro_only" if v else "unknown")
    return {"cat": cat, "vitro": v, **{k: f(terms, pt) for k, f in AXES.items()}}


def has(term, text):
    return re.search(r"\b" + re.escape(term) + r"\b", text, re.I) is not None


def logodds(a, na, b, nb):
    lo = np.log((a + .5) / (na - a + .5)) - np.log((b + .5) / (nb - b + .5))
    se = np.sqrt(1 / (a + .5) + 1 / (na - a + .5) + 1 / (b + .5) + 1 / (nb - b + .5))
    return lo, lo - 1.96 * se, lo + 1.96 * se


def boot(x):
    x = np.asarray(x, float); x = x[~np.isnan(x)]
    if not len(x):
        return np.nan, np.nan, np.nan
    bs = x[rng.integers(0, len(x), (B, len(x)))].mean(1)
    return float(x.mean()), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))


CLS = ("Is the following search query about human health, i.e. does the searcher most likely want evidence about "
       "effects in people (not animals, cells or the environment)? Query: \"{q}\"\nReply with JSON {{\"human\": true/false}}.")
llm = LLM("local", cache_name="p04_cls")
lo_rows, shift_rows, cost_rows = [], [], []
for ds in DSS:
    ids, titles, texts, qs, qrels = load(ds)
    d2p = json.load(open(f"{ROOT}/data/doc2pmid_{ds}.json"))
    info = {d: doc_info(d2p.get(d)) for d in ids}
    full = {d: f"{t} {x}" for d, t, x in zip(ids, titles, texts)}
    # ---- term log-odds: human-only vs animal-only abstracts; axis words: tagged vs untagged human abstracts
    H = [d for d in ids if info[d] and info[d]["cat"] == "human_only"]
    A = [d for d in ids if info[d] and info[d]["cat"] == "animal_only"]
    for t in sorted(set(TERMS) | {c for v in CORE.values() for c in (v if isinstance(v, tuple) else (v,))}):
        a = sum(has(t, full[d]) for d in H); b = sum(has(t, full[d]) for d in A)
        lo, l1, l2 = logodds(a, len(H), b, len(A))
        lo_rows.append({"ds": ds, "contrast": "human_vs_animal", "term": t, "p_target": a / max(len(H), 1), "p_other": b / max(len(A), 1),
                        "logodds": lo, "lo": l1, "hi": l2, "n_target": len(H), "n_other": len(A)})
    hum_docs = [d for d in ids if info[d] and info[d]["cat"] in ("human_only", "both")]
    for ax in AXES:
        tg = [d for d in hum_docs if info[d][ax]]; ut = [d for d in hum_docs if not info[d][ax]]
        core = CORE[ax]
        a = sum(has(core, full[d]) for d in tg); b = sum(has(core, full[d]) for d in ut)
        lo, l1, l2 = logodds(a, len(tg), b, len(ut))
        lo_rows.append({"ds": ds, "contrast": f"axis_{ax}", "term": core, "p_target": a / max(len(tg), 1), "p_other": b / max(len(ut), 1),
                        "logodds": lo, "lo": l1, "hi": l2, "n_target": len(tg), "n_other": len(ut)})
    # ---- human-health query flags (same classifier as analyze.py, from its cache)
    qq = json.load(open(f"{ROOT}/data/qq_{ds}.json"))
    qids = sorted(qq["queries"])
    outs = llm.chat_many([[{"role": "user", "content": CLS.format(q=qq["queries"][q])}] for q in qids], max_tokens=20, progress=False)
    hum = {q: bool((parse_json(o[0]["text"]) or {}).get("human", False)) and not ANIMAL_TXT.search(qq["queries"][q]) for q, o in zip(qids, outs)}
    R = pd.read_parquet(f"{ROOT}/results/runs_sweep_{ds}.parquet")
    for f_ in sorted(glob.glob(f"{ROOT}/results/runs_fam_*_{ds}.parquet")):  # S9 retriever families
        R = pd.concat([R, pd.read_parquet(f_)], ignore_index=True)
    R["human_q"] = R.qid.map(hum).fillna(False)
    R["info"] = R.doc.map(info)
    R = R[R["info"].notna()]  # R.info is the DataFrame method
    T10 = R[R["rank"] < 10].copy()
    T10 = T10[T10["info"].map(lambda x: x["cat"] in ("human_only", "animal_only", "both", "vitro_only"))]
    for col in ["human_only", "animal_only"]:
        T10[col] = T10["info"].map(lambda x, c=col: float(x["cat"] == c))
    T10["vitro_f"] = T10["info"].map(lambda x: float(x["vitro"]))
    for ax in AXES:
        T10[ax] = T10["info"].map(lambda x, a=ax: float(x[a]))
    metrics = ["human_only", "animal_only", "vitro_f"] + list(AXES)
    for qset in ("human", "all"):
        G = T10[T10.human_q] if qset == "human" else T10
        P = G.groupby(["retriever", "scope", "qid"])[metrics].mean()
        for m in G.retriever.unique():
            if (m, "base") not in P.index.droplevel(2):
                continue
            base = P.loc[(m, "base")]
            neutral = {k: (P.loc[(m, n)][k] - base[k]).reindex(base.index) for k in metrics for n in NEUTRAL if (m, n) in P.index.droplevel(2)}
            for s in sorted(G.scope.unique()):
                if s == "base" or (m, s) not in P.index.droplevel(2):
                    continue
                cur = P.loc[(m, s)].reindex(base.index)
                for k in metrics:
                    d = (cur[k] - base[k]).to_numpy()
                    nd = np.nanmean(np.vstack([(P.loc[(m, n)][k].reindex(base.index) - base[k]).to_numpy() for n in NEUTRAL
                                               if (m, n) in P.index.droplevel(2)]), axis=0)
                    e, lo, hi = boot(d); e2, lo2, hi2 = boot(d - nd)
                    shift_rows.append({"ds": ds, "queries": qset, "retriever": m, "scope": s, "metric": k, "base": float(base[k].mean()),
                                       "shift": e, "lo": lo, "hi": hi, "shift_minus_neutral": e2, "lo_n": lo2, "hi_n": hi2, "n_q": int(np.sum(~np.isnan(d)))})
    # ---- S3 retrieval cost, paired over queries
    rel = qq["qrels"]
    for qset in ("human", "all"):
        G = R if qset == "all" else R[R.human_q]
        for (m, s), g in G.groupby(["retriever", "scope"]):
            per = {}
            for q, gq in g.groupby("qid"):
                docs = gq.sort_values("rank").doc.tolist(); rq = rel.get(q, {})
                npos = max(1, sum(v > 0 for v in rq.values()))
                per[q] = (ndcg_at_k(docs, rq, 10), len([d for d in docs[:10] if rq.get(d, 0) > 0]) / npos,
                          len([d for d in docs[:100] if rq.get(d, 0) > 0]) / npos)
            cost_rows.append({"ds": ds, "queries": qset, "retriever": m, "scope": s, "per": per})
LO = pd.DataFrame(lo_rows); LO.to_csv(f"{ROOT}/results/term_logodds.csv", index=False)
SH = pd.DataFrame(shift_rows); SH.to_csv(f"{ROOT}/results/sweep_shifts.csv", index=False)
# cost: change vs base per query
cr = []
CB = {(r["ds"], r["queries"], r["retriever"]): r["per"] for r in cost_rows if r["scope"] == "base"}
for r in cost_rows:
    if r["scope"] == "base":
        continue
    b = CB.get((r["ds"], r["queries"], r["retriever"]), {})
    qs_ = [q for q in r["per"] if q in b]
    for i, name in enumerate(("ndcg10", "rec10", "rec100")):
        d = np.array([r["per"][q][i] - b[q][i] for q in qs_])
        e, lo, hi = boot(d)
        cr.append({"ds": r["ds"], "queries": r["queries"], "retriever": r["retriever"], "scope": r["scope"], "metric": name,
                   "base": float(np.mean([b[q][i] for q in qs_])) if qs_ else np.nan, "delta": e, "lo": lo, "hi": hi, "n_q": len(qs_)})
CO = pd.DataFrame(cr); CO.to_csv(f"{ROOT}/results/sweep_cost.csv", index=False)
# ---- S1 regression: shift (minus neutral) of the target share on the core word's log odds
fits, pts = [], []
lo_h = LO[LO.contrast == "human_vs_animal"].set_index(["ds", "term"]).logodds
for (ds, m), g in SH[(SH.queries == "human")].groupby(["ds", "retriever"]):
    for s in HUMAN_SFX + ANIMAL_SFX + VITRO_SFX:
        core = CORE[s]
        x = np.mean([lo_h.get((ds, c), np.nan) for c in (core if isinstance(core, tuple) else (core,))])
        metric = "human_only" if s in HUMAN_SFX else "animal_only" if s in ANIMAL_SFX else "vitro_f"
        row = g[(g.scope == s) & (g.metric == "human_only")]
        row_t = g[(g.scope == s) & (g.metric == metric)]
        if len(row):
            pts.append({"ds": ds, "retriever": m, "scope": s, "axis": "species", "logodds": x, "y": float(row.shift_minus_neutral.iloc[0]),
                        "y_target": float(row_t.shift_minus_neutral.iloc[0]) if len(row_t) else np.nan})
    for ax in AXES:
        row = g[(g.scope == ax) & (g.metric == ax)]
        xo = LO[(LO.ds == ds) & (LO.contrast == f"axis_{ax}")].logodds
        if len(row) and len(xo):
            pts.append({"ds": ds, "retriever": m, "scope": ax, "axis": "other", "logodds": float(xo.iloc[0]),
                        "y": np.nan, "y_target": float(row.shift_minus_neutral.iloc[0])})
PT = pd.DataFrame(pts)
for (ds, m), g in PT[PT.axis == "species"].groupby(["ds", "retriever"]):
    g = g.dropna(subset=["logodds", "y"])
    if len(g) >= 5:
        f = smf.ols("y ~ logodds", g).fit()
        fits.append({"ds": ds, "retriever": m, "slope": float(f.params.logodds), "slope_lo": float(f.conf_int().loc["logodds", 0]),
                     "slope_hi": float(f.conf_int().loc["logodds", 1]), "p": float(f.pvalues.logodds), "r2": float(f.rsquared), "n": len(g)})
FI = pd.DataFrame(fits)
PT.to_csv(f"{ROOT}/results/logodds_vs_shift_points.csv", index=False)
FI.to_csv(f"{ROOT}/results/logodds_vs_shift.csv", index=False)
print(LO[LO.contrast == "human_vs_animal"].pivot_table(index="term", columns="ds", values="logodds").round(2).to_string())
print(FI.round(3).to_string(index=False))
print(SH[(SH.queries == "human") & (SH.metric.isin(["human_only", "animal_only"]))].pivot_table(
    index=["scope", "metric"], columns=["ds", "retriever"], values="shift_minus_neutral").round(3).to_string()[:6000])
print(CO[(CO.queries == "human") & (CO.metric == "ndcg10")].pivot_table(index="scope", columns=["ds", "retriever"], values="delta").round(3).to_string()[:4000])
