"""P04 S5 analysis: do readers present animal or in-vitro findings as human evidence?

Inputs: results/s5_answers_<reader>.jsonl (Qwen3.5-9B = local, gpt-6-luna), results/s5_judged_<judge>.jsonl (main
judge, all answers; answers that cite no animal-only / in-vitro abstract are not extrapolation by definition), and
results/s5_judged_<second judge>_sub*.jsonl when present (agreement).
Per reader, dataset, retriever and retrieval condition (human-health queries):
  - no judge: share of cited abstracts that are animal-only / in vitro; share of answers citing >= 1 of them;
  - judge: unqualified extrapolation rate; the answer names a study population (judge, and a regex);
  - BioASQ correctness (judge vs the reference answer): yes 1, partial 0.5, no 0.
Each condition vs base, paired over queries (2,000 resamples, 95% CI). Extrapolation rate by the number of
animal-only studies in the top 10 (base condition).
python s5_analysis.py <main judge> [second judge]   Output: results/s5_effects.csv, results/s5_by_exposure.csv
"""

from pathlib import Path

WORKSPACE_ROOT = Path(__file__).resolve().parents[1].as_posix()
import glob
import json
import sys

import numpy as np
import pandas as pd

ROOT = WORKSPACE_ROOT
JM = sys.argv[1]
J2 = sys.argv[2] if len(sys.argv) > 2 else None
rng = np.random.default_rng(0)
NONHUMAN = ("animal_only", "vitro_only")
A = []
for f in sorted(glob.glob(f"{ROOT}/results/s5_answers_*.jsonl")):
    A += [json.loads(l) for l in open(f)]
A = pd.DataFrame(A)
A = A[~A.error.astype(bool) & (A.answer.str.len() > 0)].copy()
KEY = ["ds", "retriever", "cond", "qid", "reader"]
J = pd.DataFrame([json.loads(l) for l in open(f"{ROOT}/results/s5_judged_{JM}.jsonl")])
D = A.merge(J[KEY + ["extrap", "states_pop", "states_pop_regex", "correct"]], on=KEY, how="left")
D["n_cited"] = D.cited.map(len)
D["cited_nonhuman"] = [np.mean([c[i - 1] in NONHUMAN for i in ci]) if len(ci) else np.nan for c, ci in zip(D.cats, D.cited)]
D["cites_nonhuman"] = [float(any(c[i - 1] in NONHUMAN for i in ci)) for c, ci in zip(D.cats, D.cited)]
D["extrap"] = D.extrap.map(lambda x: float(x is True))
D["states_pop_regex"] = D.states_pop_regex.astype(float)
D["correct_s"] = D.correct.map({"yes": 1.0, "partial": 0.5, "no": 0.0})
D["top10_animal"] = D.cats.map(lambda c: sum(x == "animal_only" for x in c))
MET = ["cited_nonhuman", "cites_nonhuman", "extrap", "states_pop_regex", "correct_s"]


def boot(x):
    x = np.asarray(x, float); x = x[~np.isnan(x)]
    if not len(x):
        return [np.nan, np.nan, np.nan]
    bs = x[rng.integers(0, len(x), (2000, len(x)))].mean(1)
    return [float(x.mean()), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]


rows = []
for (rd, ds, ret), g in D.groupby(["reader", "ds", "retriever"]):
    base = g[g.cond == "base"].set_index("qid")
    for cond, h in g.groupby("cond"):
        h = h.set_index("qid")
        r = {"reader": rd, "ds": ds, "retriever": ret, "cond": cond, "n": len(h)}
        for m in MET:
            r[m], r[f"{m}_lo"], r[f"{m}_hi"] = boot(h[m])
            if cond != "base":
                q = h.index.intersection(base.index)
                r[f"d_{m}"], r[f"d_{m}_lo"], r[f"d_{m}_hi"] = boot((h.loc[q, m] - base.loc[q, m]).values)
        rows.append(r)
E = pd.DataFrame(rows)
E.to_csv(f"{ROOT}/results/s5_effects.csv", index=False)
X = D[D.cond == "base"].assign(k=lambda d: d.top10_animal.clip(upper=3))
BX = X.groupby(["reader", "k"]).agg(n=("extrap", "size"), extrap=("extrap", "mean"), cites_nonhuman=("cites_nonhuman", "mean"),
                                    states_pop_regex=("states_pop_regex", "mean")).reset_index()
BX.to_csv(f"{ROOT}/results/s5_by_exposure.csv", index=False)
pd.set_option("display.width", 250); pd.set_option("display.max_rows", 200)
print(f"answers {len(D)}  judged {JM}: extrapolation among answers citing non-human abstracts "
      f"{D[D.cites_nonhuman == 1].extrap.mean():.3f} (n {int(D.cites_nonhuman.sum())})")
cols = ["reader", "ds", "retriever", "cond", "n", "cited_nonhuman", "cites_nonhuman", "extrap", "extrap_lo", "extrap_hi",
        "d_extrap", "d_extrap_lo", "d_extrap_hi", "states_pop_regex", "correct_s", "d_correct_s", "d_correct_s_lo", "d_correct_s_hi"]
print(E[[c for c in cols if c in E.columns]].round(3).to_string(index=False))
print("\nextrapolation by animal-only studies in the top 10 (base condition; 3 = 3 or more)")
print(BX.round(3).to_string(index=False))
if J2:
    for f in sorted(glob.glob(f"{ROOT}/results/s5_judged_{J2}_sub*.jsonl")):
        S = pd.DataFrame([json.loads(l) for l in open(f)])
        S = S[S.judged].merge(J[KEY + ["extrap"]].rename(columns={"extrap": "main"}), on=KEY)
        a, b = S.extrap.map(lambda x: x is True), S.main.map(lambda x: x is True)
        po = float((a == b).mean()); pe = float(a.mean() * b.mean() + (1 - a.mean()) * (1 - b.mean()))
        print(f"\nagreement {J2} vs {JM} on {len(S)} judged answers: {po:.3f}, kappa {(po - pe) / (1 - pe):.3f}; "
              f"extrapolation rate {a.mean():.3f} vs {b.mean():.3f}")
