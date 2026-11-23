"""P04 S6: are the results an artefact of MeSH labelling errors?

MEDLINE assigns MeSH by hand, by an automated indexer, or automated-then-curated (IndexingMethod in the PubMed
XML; data/pubmed_indexing.jsonl from fetch_pubmed.py). Automated indexing is reported to miss "Humans" more often.
Main readouts recomputed on four document labellings:
  all          every MeSH-labelled document (as in the main analysis)
  manual       only documents indexed by hand (MEDLINE status, no IndexingMethod attribute); others unlabelled
  by year      publication year <= 2009, 2010-2014, 2015-2019, >= 2020
  text_agree   only documents whose text classification (S7, results/s7_doc_labels.parquet) agrees with MeSH
Readouts (human-health queries, base query): top-10 human-only and animal-only share; exposure (queries with
>= 1 animal-only study in the top 10); shifts for "in humans", "in mice", "in patients" (query bootstrap CIs).
Runs: results/runs_sweep_<ds>.parquet when present, else runs_<ds> + runs_new_<ds> + runs_extra_<ds>_*.
Output: results/s6_robustness.csv
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

sys.path.insert(0, f"{WORKSPACE_ROOT}/shared")
from llm import LLM, parse_json  # noqa: E402

ROOT = WORKSPACE_ROOT
rng = np.random.default_rng(0)
VITRO = re.compile(r"Cell Line|Cells, Cultured|In Vitro Techniques|Cell Culture", re.I)
ANIMAL_TXT = re.compile(r"\b(mice|mouse|murine|rats?|rodents?|rabbits?|dogs?|pigs?|primates?|monkeys?|zebrafish|drosophila|in vitro|cell lines?|cultured cells)\b", re.I)
IDX = {j["pmid"]: j for j in (json.loads(l) for l in open(f"{ROOT}/data/pubmed_indexing.jsonl"))}
mesh = {json.loads(l)["pmid"]: json.loads(l) for l in open(f"{ROOT}/data/mesh.jsonl")}
TXT = None
if os.path.exists(f"{ROOT}/results/s7_doc_labels.parquet"):
    TXT = pd.read_parquet(f"{ROOT}/results/s7_doc_labels.parquet").set_index("pmid")["text_label"].to_dict()


def cat(pmid):
    r = mesh.get(pmid)
    if not r or not r["mesh"]:
        return "unknown"
    t = {m.split("/")[0].strip("* ") for m in r["mesh"]}
    h, a = "Humans" in t, "Animals" in t
    return "both" if h and a else "human_only" if h else "animal_only" if a else ("vitro_only" if any(VITRO.search(x) for x in t) else "unknown")


def subset_ok(pmid, name):
    j = IDX.get(pmid, {})
    if name == "all":
        return True
    if name == "manual":
        return j.get("indexing") == "Manual" and j.get("status") == "MEDLINE"
    if name.startswith("year"):
        y = j.get("year") or (mesh.get(pmid) or {}).get("year")
        if not y:
            return False
        lo, hi = {"year<=2009": (0, 2009), "year2010-14": (2010, 2014), "year2015-19": (2015, 2019), "year>=2020": (2020, 9999)}[name]
        return lo <= y <= hi
    if name == "text_agree":
        return TXT is not None and TXT.get(pmid) == cat(pmid)
    return True


def boot(x):
    x = np.asarray(x, float); x = x[~np.isnan(x)]
    if not len(x):
        return np.nan, np.nan, np.nan
    bs = x[rng.integers(0, len(x), (1000, len(x)))].mean(1)
    return float(x.mean()), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))


CLS = ("Is the following search query about human health, i.e. does the searcher most likely want evidence about "
       "effects in people (not animals, cells or the environment)? Query: \"{q}\"\nReply with JSON {{\"human\": true/false}}.")
llm = LLM("local", cache_name="p04_cls")
rows = []
for ds in ["bioasq", "nfcorpus", "trec-covid"]:
    if os.path.exists(f"{ROOT}/results/runs_sweep_{ds}.parquet"):
        R = pd.read_parquet(f"{ROOT}/results/runs_sweep_{ds}.parquet")
    else:
        R = pd.concat([pd.read_parquet(f) for f in [f"{ROOT}/results/runs_{ds}.parquet", f"{ROOT}/results/runs_new_{ds}.parquet"]
                       + sorted(glob.glob(f"{ROOT}/results/runs_extra_{ds}_*.parquet")) if os.path.exists(f)])
    R = R[(R["rank"] < 10) & R.scope.isin(["base", "humans", "mice", "patients"])]
    d2p = json.load(open(f"{ROOT}/data/doc2pmid_{ds}.json"))
    qq = json.load(open(f"{ROOT}/data/qq_{ds}.json"))
    qids = sorted(qq["queries"])
    outs = llm.chat_many([[{"role": "user", "content": CLS.format(q=qq["queries"][q])}] for q in qids], max_tokens=20, progress=False)
    hum = {q for q, o in zip(qids, outs) if bool((parse_json(o[0]["text"]) or {}).get("human", False)) and not ANIMAL_TXT.search(qq["queries"][q])}
    R = R[R.qid.isin(hum)].copy()
    R["pmid"] = R.doc.map(d2p)
    R["cat"] = R.pmid.map(lambda p: cat(p) if isinstance(p, str) else "unknown")
    for name in ["all", "manual", "year<=2009", "year2010-14", "year2015-19", "year>=2020", "text_agree"]:
        ok = R.pmid.map(lambda p: isinstance(p, str) and subset_ok(p, name))
        G = R[ok & R.cat.isin(["human_only", "animal_only", "both", "vitro_only"])]
        if G.empty:
            continue
        G = G.assign(h=(G.cat == "human_only").astype(float), a=(G.cat == "animal_only").astype(float))
        for m, g in G.groupby("retriever"):
            P = g.groupby(["qid", "scope"])[["h", "a"]].mean().unstack("scope")
            cnt = g[g.scope == "base"].groupby("qid").a.sum()
            row = {"ds": ds, "retriever": m, "labels": name, "n_q": int(P.shape[0]), "labelled_docs_top10": int((g.scope == "base").sum())}
            for met in ("h", "a"):
                if (met, "base") in P.columns:
                    row[f"{met}_base"] = float(P[(met, "base")].mean())
            row["exposure"], row["exposure_lo"], row["exposure_hi"] = boot((cnt >= 1).astype(float))
            for sc, met in (("humans", "h"), ("patients", "h"), ("mice", "a")):
                if (met, sc) in P.columns and (met, "base") in P.columns:
                    row[f"{met}_shift_{sc}"], row[f"{met}_shift_{sc}_lo"], row[f"{met}_shift_{sc}_hi"] = boot((P[(met, sc)] - P[(met, "base")]).to_numpy())
            rows.append(row)
D = pd.DataFrame(rows)
D.to_csv(f"{ROOT}/results/s6_robustness.csv", index=False)
pd.set_option("display.width", 250)
print(D[["ds", "retriever", "labels", "n_q", "h_base", "a_base", "exposure", "h_shift_humans", "h_shift_patients", "a_shift_mice"]].round(3).to_string(index=False))
