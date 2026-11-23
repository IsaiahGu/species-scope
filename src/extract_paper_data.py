"""P04: build the compact writing pack paper_data/ from results/ (stdlib + numpy/scipy; no model calls, no parquet).

python -I extract_paper_data.py <project dir> <check log path>
Top level of paper_data/: one file per figure or table of the writing guide. supporting/: everything the text quotes.
"""
import csv
import json
import os
import re
import sys
from collections import defaultdict

import numpy as np
from scipy import stats

ROOT = sys.argv[1]
R = ROOT + "/results"
O = ROOT + "/paper_data"
os.makedirs(O + "/supporting", exist_ok=True)
LOG = open(sys.argv[2], "w", encoding="utf-8")
rng = np.random.default_rng(0)   # same seed and call order as guide_checks_3, so the intervals are identical

DSS = ["bioasq", "nfcorpus", "trec-covid"]
RETR = [("bm25", "bm25", "BM25", "lexical"), ("splade-pp", "splade_pp", "SPLADE++", "learned sparse"),
        ("gte-moderncolbert", "gte_moderncolbert", "GTE-ModernColBERT", "multi-vector"),
        ("e5", "e5", "e5-base-v2", "general dense"), ("qwen3-emb", "qwen3_emb", "Qwen3-Embedding-0.6B", "general dense"),
        ("qwen3-emb-4b", "qwen3_emb_4b", "Qwen3-Embedding-4B", "general dense"), ("egemma2", "egemma2", "EmbeddingGemma 2", "general dense"),
        ("harrier", "harrier", "harrier-oss-v1-0.6b", "general dense"), ("medcpt", "medcpt", "MedCPT", "domain dense"),
        ("medcpt+ce", "medcpt_ce", "MedCPT + cross-encoder", "reranking"), ("rrf(bm25,qwen3-emb)", "rrf", "RRF(BM25, Qwen3-Embedding-0.6B)", "fusion")]
RKEYS = [r[0] for r in RETR]
RNAME = {r[0]: r[2] for r in RETR}
RFAM = {r[0]: r[3] for r in RETR}
PHRASE = {"humans": "in humans", "people": "in people", "patients": "in patients", "adults": "in adults", "participants": "in human participants",
          "subjects": "in human subjects", "volunteers": "in volunteers", "clinical_trial": "in a clinical trial",
          "rct": "in a randomized controlled trial", "men_women": "in men and women", "mice": "in mice", "rats": "in rats",
          "animals": "in animals", "animal_models": "in animal models", "rodents": "in rodents", "vitro": "in vitro",
          "cell_lines": "in cell lines", "general": "in general", "detail": "in detail", "literature": "in the literature",
          "children": "in children", "older_adults": "in older adults", "women": "in women", "men": "in men", "pregnancy": "in pregnancy",
          "rct_only": "randomized controlled trial", "case_report": "case report", "review": "review"}
CORE = {"humans": "humans", "people": "people", "patients": "patients", "adults": "adults", "participants": "participants", "subjects": "subjects",
        "volunteers": "volunteers", "clinical_trial": "clinical trial", "rct": "randomized controlled trial", "men_women": "men+women",
        "mice": "mice", "rats": "rats", "animals": "animals", "animal_models": "animal models", "rodents": "rodents", "vitro": "in vitro",
        "cell_lines": "cell lines"}
HUMAN = ["humans", "people", "patients", "adults", "participants", "subjects", "volunteers", "clinical_trial", "rct", "men_women"]
ANIMAL = ["mice", "rats", "animals", "animal_models", "rodents"]
VITRO = ["vitro", "cell_lines"]
NEUTRAL = ["general", "detail", "literature"]
AXES = ["adults", "children", "older_adults", "women", "men", "pregnancy", "rct_only", "case_report", "review"]
GROUP = {**{s: "human" for s in HUMAN}, **{s: "animal" for s in ANIMAL}, **{s: "in vitro" for s in VITRO}, **{s: "neutral" for s in NEUTRAL}}
FIG3 = ["humans", "human", "people", "patients", "participants", "mice", "rats", "animals"]
NONHUMAN = ("animal_only", "vitro_only")
READERS = [("gpt-6-luna", "gpt-6-luna"), ("local", "Qwen3.5-9B")]


def log(*a):
    LOG.write(" ".join(str(x) for x in a) + "\n")


def rd(name):
    with open(f"{R}/{name}", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def fm(v, nd=4):
    if v is None or (isinstance(v, float) and (np.isnan(v) or np.isinf(v))):
        return ""
    if isinstance(v, (int, np.integer)):
        return str(int(v))
    if isinstance(v, (float, np.floating)):
        return f"{v:.2e}" if 0 < abs(v) < 1e-4 else f"{v:.{nd}f}"
    return str(v)


def wr(name, header, rows):
    with open(f"{O}/{name}", "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        for r in rows:
            w.writerow([fm(x) for x in r])
    log(f"wrote {name}: {len(rows)} rows")


def tidy(x):
    try:
        return int(x)
    except ValueError:
        pass
    try:
        return float(x)
    except ValueError:
        return x


def fl(x):
    v = tidy(x)
    return None if isinstance(v, str) else v


def copy_rounded(src, dst, keep=lambda r: True):
    rows = [r for r in rd(src) if keep(r)]
    wr(dst, list(rows[0].keys()), [[tidy(v) for v in r.values()] for r in rows])


def mean(v):
    return float(np.mean(v)) if len(v) else float("nan")


# ------------------------------------------------------------------ sweep: shifts, log-odds, cost
SH = [r for r in rd("sweep_shifts.csv") if r["queries"] == "human"]
sh = {(r["ds"], r["retriever"], r["scope"], r["metric"]): r for r in SH}
PT = [r for r in rd("logodds_vs_shift_points.csv") if r["axis"] == "species"]
lod = {(r["ds"], r["scope"]): float(r["logodds"]) for r in PT}
LO = rd("term_logodds.csv")
lo_t = {(r["ds"], r["term"]): r for r in LO if r["contrast"] == "human_vs_animal"}
CO = [r for r in rd("sweep_cost.csv") if r["queries"] == "human"]
co = {(r["ds"], r["retriever"], r["scope"], r["metric"]): r for r in CO}


def sig(r):
    return 1 if float(r["lo_n"]) > 0 else -1 if float(r["hi_n"]) < 0 else 0


def cells(scope, metric, dss=DSS):
    return [sh[(ds, m, scope, metric)] for ds in dss for m in RKEYS if (ds, m, scope, metric) in sh]


ORDER17 = sorted(HUMAN + ANIMAL + VITRO, key=lambda s: -lod[("bioasq", s)])
rows = []
for i, s in enumerate(ORDER17):
    t = lo_t.get(("bioasq", CORE[s]))
    row = [i + 1, s, PHRASE[s], GROUP[s], CORE[s], lod[("bioasq", s)], lod[("nfcorpus", s)], lod[("trec-covid", s)],
           float(t["p_target"]) if t else None, float(t["p_other"]) if t else None]
    for m in RKEYS:
        r = sh[("bioasq", m, s, "human_only")]
        row += [float(r["shift_minus_neutral"]), sig(r)]
    for ds in DSS:
        row.append(mean([float(r["shift_minus_neutral"]) for r in cells(s, "human_only", [ds])]))
    allc = cells(s, "human_only")
    row += [sum(sig(r) < 0 for r in allc), sum(sig(r) > 0 for r in allc), len(allc),
            mean([float(co[("bioasq", m, s, "ndcg10")]["delta"]) for m in RKEYS])]
    rows.append(row)
hdr = ["row", "scope", "phrase", "group", "core_term", "logodds_bioasq", "logodds_nfcorpus", "logodds_trec_covid", "p_human_only_bioasq", "p_animal_only_bioasq"]
for k, col, _, _ in RETR:
    hdr += [col, col + "_sig"]
hdr += ["mean_bioasq", "mean_nfcorpus", "mean_trec_covid", "cells_sig_neg", "cells_sig_pos", "cells", "ndcg10_change_bioasq"]
wr("fig2_heatmap.csv", hdr, rows)
FIG2 = {r[1]: dict(zip(hdr, r)) for r in rows}

rows = []
for s in ORDER17 + NEUTRAL:
    for metric in ["human_only", "animal_only", "vitro_f"]:
        for ds in DSS:
            for m in RKEYS:
                r = sh.get((ds, m, s, metric))
                if r:
                    rows.append([s, PHRASE[s], metric, ds, m, float(r["base"]), float(r["shift"]), float(r["shift_minus_neutral"]),
                                 float(r["lo_n"]), float(r["hi_n"]), sig(r), int(r["n_q"])])
wr("supporting/shifts_species.csv", ["scope", "phrase", "share", "ds", "retriever", "base", "shift_raw", "shift_minus_neutral", "lo", "hi", "sig", "n_q"], rows)

# summary by suffix x share x dataset (means over the 11 retrievers, counts of significant cells)
rows = []
for s in ORDER17 + NEUTRAL:
    for metric in ["human_only", "animal_only", "vitro_f"]:
        for ds in DSS:
            c = cells(s, metric, [ds])
            if c:
                v = [float(r["shift_minus_neutral"]) for r in c]
                rows.append([s, PHRASE[s], metric, ds, mean([float(r["base"]) for r in c]), mean(v), min(v), max(v),
                             sum(sig(r) > 0 for r in c), sum(sig(r) < 0 for r in c), len(c)])
wr("supporting/shifts_species_summary.csv", ["scope", "phrase", "share", "ds", "base_mean", "shift_mean", "shift_min", "shift_max", "n_sig_pos", "n_sig_neg", "n_retrievers"], rows)
SUMM = {(r[0], r[2], r[3]): r for r in rows}

# implied change of the share indexed with both Humans and Animals under "in humans"
rows = []
for ds in DSS:
    v = [-(float(sh[(ds, m, "humans", "human_only")]["shift_minus_neutral"]) + float(sh[(ds, m, "humans", "animal_only")]["shift_minus_neutral"])) for m in RKEYS]
    rows.append([ds, mean(v), min(v), max(v), sum(x > 0 for x in v), len(v)])
wr("supporting/in_humans_both_share.csv", ["ds", "implied_shift_both_mean", "min", "max", "n_positive", "n_retrievers"], rows)
BOTH = {r[0]: r for r in rows}

# other population axes
rows = []
for ds in DSS:
    for a in AXES:
        c = cells(a, a, [ds])
        if not c:
            continue
        b = np.array([float(r["base"]) for r in c]); s_ = np.array([float(r["shift_minus_neutral"]) for r in c]); raw = np.array([float(r["shift"]) for r in c])
        ok = b > 0
        hum = [float(r["shift_minus_neutral"]) for r in cells(a, "human_only", [ds])]
        ani = [float(r["shift_minus_neutral"]) for r in cells(a, "animal_only", [ds])]
        rows.append([a, PHRASE[a], ds, b.mean(), s_.mean(), s_.min(), s_.max(), sum(sig(r) > 0 for r in c), len(c),
                     float(((b[ok] + raw[ok]) / b[ok]).mean()) if ok.any() else None, mean(hum), mean(ani)])
wr("supporting/other_axes.csv", ["axis", "phrase", "ds", "base_share_mean", "own_share_shift_mean", "min", "max", "n_sig_pos", "n_retrievers",
                                 "ratio_after_before", "human_only_shift_mean", "animal_only_shift_mean"], rows)
AX = {(r[0], r[2]): r for r in rows}

# retrieval cost of the suffixes (computed on MeSH-labelled documents only: provisional)
rows = []
for s in ORDER17 + NEUTRAL + [a for a in AXES if a != "adults"]:
    for ds in DSS:
        c = [co[(ds, m, s, "ndcg10")] for m in RKEYS if (ds, m, s, "ndcg10") in co]
        v = [float(r["delta"]) for r in c]
        if not v:
            continue
        rows.append([s, PHRASE[s], ds, mean(v), min(v), max(v), sum(float(r["hi"]) < 0 for r in c), len(c),
                     mean([float(co[(ds, m, s, "rec10")]["delta"]) for m in RKEYS if (ds, m, s, "rec10") in co])])
wr("supporting/suffix_cost.csv", ["scope", "phrase", "ds", "ndcg10_change_mean", "min", "max", "n_sig_neg", "n_retrievers", "recall10_change_mean"], rows)
COST = {(r[0], r[2]): r for r in rows}

# word prevalence
rows = []
for r in LO:
    f3 = int(r["contrast"] == "human_vs_animal" and r["term"] in FIG3)
    rows.append([f3, FIG3.index(r["term"]) + 1 if f3 else None, r["ds"], r["contrast"], r["term"], float(r["p_target"]), float(r["p_other"]),
                 float(r["logodds"]), float(r["lo"]), float(r["hi"]), int(r["n_target"]), int(r["n_other"])])
rows.sort(key=lambda x: (-x[0], x[1] or 0, DSS.index(x[2]), x[3], x[4]))
wr("fig3_word_prevalence.csv", ["in_fig3", "fig3_order", "ds", "contrast", "term", "p_in_target", "p_in_other", "logodds", "lo", "hi", "n_target", "n_other"], rows)

# direction rule and within-group correlations
agree = defaultdict(lambda: [0, 0])
for r in PT:
    ok = int(np.sign(float(r["logodds"])) == np.sign(float(r["y"])))
    for key in ("ALL", "group: " + GROUP[r["scope"]], r["scope"]):
        agree[key][0] += ok; agree[key][1] += 1
rows = [[k, PHRASE.get(k, ""), lod.get(("bioasq", k)), agree[k][0], agree[k][1], agree[k][0] / agree[k][1]]
        for k in ["ALL", "group: human", "group: animal", "group: in vitro"] + ORDER17]
wr("supporting/logodds_sign_agreement.csv", ["scope", "phrase", "logodds_bioasq", "cells_same_sign", "cells", "share"], rows)
AGREE = dict(agree)


def corr(pts):
    x = np.array([a for a, b in pts]); y = np.array([b for a, b in pts])
    return float(stats.pearsonr(x, y)[0]), float(stats.spearmanr(x, y)[0])


fit = {(r["ds"], r["retriever"]): r for r in rd("logodds_vs_shift.csv")}
pts = defaultdict(dict)
for r in PT:
    pts[(r["ds"], r["retriever"])][r["scope"]] = (float(r["logodds"]), float(r["y"]))
rows = []
for ds in DSS:
    for m in RKEYS:
        d = pts[(ds, m)]
        g = {"all": HUMAN + ANIMAL + VITRO, "h10": HUMAN, "h9": [s for s in HUMAN if s != "humans"], "n7": ANIMAL + VITRO}
        c = {k: corr([d[s] for s in v]) for k, v in g.items()}
        rk = [s for _, s in sorted((d[s][1], s) for s in HUMAN)].index("humans") + 1
        f = fit[(ds, m)]
        rows.append([ds, m, float(f["slope"]), float(f["slope_lo"]), float(f["slope_hi"]), float(f["r2"]), float(f["p"]),
                     c["all"][0], c["h10"][0], c["h10"][1], c["h9"][0], c["h9"][1], c["n7"][0], c["n7"][1], rk])
med = ["median", "33 cells"] + [float(np.median([r[i] for r in rows])) for i in range(2, 14)] + [None]
npos = ["n_positive", "33 cells"] + [int(sum(r[i] > 0 for r in rows)) for i in range(2, 14)] + [int(sum(r[14] == 1 for r in rows))]
CORR = {"med": med, "npos": npos}
wr("supporting/logodds_fits.csv", ["ds", "retriever", "slope_all17", "slope_lo", "slope_hi", "r2_all17", "p", "r_all17", "r_human10", "rho_human10",
                                   "r_human9_without_humans", "rho_human9", "r_nonhuman7", "rho_nonhuman7", "rank_of_humans_among_human10"], rows + [med, npos])

# ------------------------------------------------------------------ tables 1 and 2
qc = {r["ds"]: r for r in rd("query_classification.csv")}
comp = {(r["ds"], r["retriever"]): r for r in rd("top10_composition.csv")}
rows = []
for ds in DSS:
    try:
        d2p = json.load(open(f"{ROOT}/data/doc2pmid_{ds}.json", encoding="utf-8"))
        ndoc, npm = len(d2p), sum(bool(v) for v in d2p.values())
    except OSError:
        ndoc = npm = None
    t = lo_t[(ds, "humans")]
    pool = comp[(ds, "judged-relevant pool")]
    nq = [int(r["n_q"]) for r in SH if r["ds"] == ds and r["metric"] == "human_only"]
    rows.append([ds, ndoc, npm, int(qc[ds]["n_queries"]), int(round(float(qc[ds]["human_share"]) * int(qc[ds]["n_queries"]))), min(nq), max(nq),
                 float(qc[ds]["human_share"]),
                 float(qc[ds]["agreement_gpt4o_mini"]), float(qc[ds]["agreement_gpt6_luna"]), int(t["n_target"]), int(t["n_other"]),
                 float(pool["human_only"]), float(pool["animal_only"]), float(pool["both"]), float(pool["vitro_only"]),
                 mean([float(sh[(ds, m, "humans", "human_only")]["base"]) for m in RKEYS]),
                 mean([float(sh[(ds, m, "humans", "animal_only")]["base"]) for m in RKEYS])])
wr("tab1_data.csv", ["ds", "docs_in_id_map", "docs_with_pmid", "queries", "human_health_queries", "queries_scored_min", "queries_scored_max",
                     "human_health_share", "agreement_gpt4o_mini",
                     "agreement_gpt6_luna", "human_only_abstracts", "animal_only_abstracts", "pool_human_only", "pool_animal_only", "pool_both",
                     "pool_vitro_only", "top10_human_only_mean11", "top10_animal_only_mean11"], rows)
TAB1 = {r[0]: r for r in rows}
copy_rounded("s6_robustness.csv", "supporting/s6_robustness.csv")

ex = {(r["ds"], r["retriever"]): r for r in rd("exposure.csv")}
rows = []
for ds in DSS:
    for m in RKEYS:
        e = ex.get((ds, m))
        base_a = float(sh[(ds, m, "humans", "animal_only")]["base"]); base_h = float(sh[(ds, m, "humans", "human_only")]["base"])
        rows.append([ds, m, RNAME[m], RFAM[m], base_h, base_a] + ([fl(e[c]) for c in (
            "n_queries", "q_with_1plus_animal_only", "q1_lo", "q1_hi", "q_with_3plus_animal_only", "animal_only_hits_judged_relevant", "n_animal_hits",
            "animal_excess_vs_relevant_pool", "excess_lo", "excess_hi")] if e else [None] * 10))
wr("tab2_exposure.csv", ["ds", "retriever", "name", "family", "top10_human_only", "top10_animal_only", "n_queries", "exposure", "exposure_lo", "exposure_hi",
                         "exposure_3plus", "animal_hits_judged_relevant", "n_animal_hits", "excess_vs_pool", "excess_lo", "excess_hi"], rows)
TAB2 = {(r[0], r[1]): r for r in rows}
copy_rounded("s8_scale.csv", "supporting/s8_corpus_size.csv")
copy_rounded("s4_instructions.csv", "supporting/s4_instructions.csv")
copy_rounded("within_relevant_gap.csv", "supporting/within_relevant_gap.csv")
wr("supporting/retrievers.csv", ["key", "column_in_fig2", "name", "family", "has_exposure_row"], [[k, c, n, f, int(("bioasq", k) in ex)] for k, c, n, f in RETR])

fx = defaultdict(dict)
for r in rd("fix_mesh.csv"):
    fx[(r["ds"], r["retriever"])][r["fix"]] = r
rows = []
for (ds, m), d in fx.items():
    for name in ["mesh_demote", "text_demote"]:
        if name in d and "none" in d:
            a0 = float(d["none"]["animal10"]); a1 = float(d[name]["animal10"])
            rows.append([ds, m, name, float(d["none"]["ndcg10"]), float(d[name]["ndcg10"]) - float(d["none"]["ndcg10"]),
                         float(d[name]["rec10"]) - float(d["none"]["rec10"]), a0, a1 - a0, (a0 - a1) / a0 if a0 > 0 else None])
wr("supporting/fix_demotion.csv", ["ds", "retriever", "fix", "ndcg10_base", "ndcg10_change", "recall10_change", "animal_only_top10_base",
                                   "animal_only_change", "share_removed"], rows)
FIX = rows

S7 = json.load(open(f"{R}/s7_classifier.json", encoding="utf-8"))
rows = []
for meth in ["regex", "bow", "llm"]:
    for cls in ["human_only", "animal_only", "both", "vitro_only", "neither"]:
        c = S7[meth][cls]
        rows.append([meth, cls, c["precision"], c["recall"], c["f1-score"], int(c["support"]), S7[meth]["accuracy"], S7["n_train"], S7["n_test"]])
wr("supporting/s7_classifiers.csv", ["method", "class", "precision", "recall", "f1", "support", "accuracy", "n_train", "n_test"], rows)

rows = []
for fn in sorted(os.listdir(R)):
    if fn.startswith("s5_judge_control") and fn.endswith(".json"):
        d = json.load(open(f"{R}/{fn}", encoding="utf-8"))
        rows.append([d["judge"], "v2" if "_v2" in fn else "v1", 2 if "control2" in fn else 1, d["neg_human"], d["neg_qualified"], d["pos_unqualified"],
                     d["unparsed"], int(d["pass"]), fn])
wr("supporting/s5_judge_control.csv", ["judge", "judge_prompt", "control_set", "acc_human_only_answers", "acc_qualified_answers", "acc_unqualified_answers",
                                       "unparsed", "passes_0.90_gate", "source_file"], rows)

# ------------------------------------------------------------------ answers (S5)
BRACKET = re.compile(r"\[([\d\s,;\-–]+)\]")


def brackets(ans, n):
    """Every citation bracket as a list of abstract numbers ("[1, 2, 8]", "[2-4]" included; answers.py only read "[5]")."""
    out = []
    for g in BRACKET.findall(ans):
        nums = []
        for part in re.split(r"[,;]", g):
            part = part.strip()
            m = re.fullmatch(r"(\d{1,2})\s*[\-–]\s*(\d{1,2})", part)
            if m:
                nums += list(range(int(m.group(1)), int(m.group(2)) + 1))
            elif re.fullmatch(r"\d{1,2}", part):
                nums.append(int(part))
        nums = [x for x in nums if 1 <= x <= n]
        if nums:
            out.append(nums)
    return out


A = {}
for rdr, fn in [("gpt-6-luna", "s5_answers_gpt-6-luna.jsonl"), ("local", "s5_answers_local.jsonl")]:
    for l in open(f"{R}/{fn}", encoding="utf-8"):
        a = json.loads(l)
        g = brackets(a["answer"], len(a["docs"]))
        a["full"] = sorted({x for grp in g for x in grp})
        a["any_nh"] = any(a["cats"][c - 1] in NONHUMAN for c in a["full"])
        a["sole"] = any(all(a["cats"][c - 1] in NONHUMAN for c in grp) for grp in g)   # a statement whose only cited support is non-human
        A[(a["ds"], a["retriever"], a["cond"], a["qid"], a["reader"])] = a
J = [json.loads(l) for l in open(f"{R}/s5_judged_gpt-6-luna_v2.jsonl", encoding="utf-8")]
CS = {"yes": 1.0, "partial": 0.5, "no": 0.0}
D = {}
for j in J:
    cn = [j["cats"][c - 1] for c in j["cited"]]
    j["cites_nh"] = any(c in NONHUMAN for c in cn)
    a = A[(j["ds"], j["retriever"], j["cond"], j["qid"], j["reader"])]
    j["unjudged"] = a["sole"] and not j["cites_nh"]      # rests a statement on non-human abstracts only, but was never sent to the judge
    j["any_nh"] = a["any_nh"]
    j["k"] = sum(c == "animal_only" for c in j["cats"])
    j["ex"] = j["extrap"] is True
    D[(j["reader"], j["ds"], j["retriever"], j["cond"], j["qid"])] = {"ex": float(j["ex"]), "cites": float(j["cites_nh"]), "k": j["k"],
                                                                     "corr": CS.get(j["correct"], np.nan)}
log(f"S5 judged rows {len(J)}; cite a non-human abstract {sum(j['cites_nh'] for j in J)}; sent to judge {sum(bool(j['judged']) for j in J)}; "
    f"extrapolation {sum(j['ex'] for j in J)}; extrapolation without a non-human citation {sum(j['ex'] and not j['cites_nh'] for j in J)}")
CONDS = ["labelled", "patients", "rerank", "mesh_demote"]
RD = [r for r, _ in READERS]


def cluster_boot(by, B=2000):
    keys = list(by)
    sums = np.array([np.nansum(by[k]) for k in keys]); ns = np.array([np.sum(~np.isnan(by[k])) for k in keys])
    idx = rng.integers(0, len(keys), (B, len(keys)))
    bs = sums[idx].sum(1) / np.maximum(ns[idx].sum(1), 1)
    return float(sums.sum() / ns.sum()), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5)), int(ns.sum())


def pooled(metric, rdr, cond, dss, level=False):
    cl = defaultdict(list)
    for (r, ds, ret, c, q), v in D.items():
        if r != rdr or c != cond or ds not in dss:
            continue
        if level:
            cl[(ds, q)].append(v[metric])
        else:
            b = D.get((r, ds, ret, "base", q))
            if b is not None:
                cl[(ds, q)].append(v[metric] - b[metric])
    return cluster_boot(cl)


P = []   # (datasets, reader, condition, quantity, kind, est, lo, hi, n)
for dss, name in [(("bioasq", "nfcorpus"), "bioasq+nfcorpus"), (("bioasq",), "bioasq"), (("nfcorpus",), "nfcorpus")]:
    for rdr in RD:
        P.append([name, rdr, "base", "extrapolation", "level", *pooled("ex", rdr, "base", dss, level=True)])
        P.append([name, rdr, "base", "cites_nonhuman", "level", *pooled("cites", rdr, "base", dss, level=True)])
        for cond in CONDS:
            P.append([name, rdr, cond, "extrapolation", "change_vs_base", *pooled("ex", rdr, cond, dss)])
            P.append([name, rdr, cond, "cites_nonhuman", "change_vs_base", *pooled("cites", rdr, cond, dss)])
for rdr in RD:
    P.append(["bioasq", rdr, "base", "correctness", "level", *pooled("corr", rdr, "base", ("bioasq",), level=True)])
    for cond in CONDS:
        P.append(["bioasq", rdr, cond, "correctness", "change_vs_base", *pooled("corr", rdr, cond, ("bioasq",))])
for rdr in RD:
    for cond in ["base", "labelled", "patients", "rerank"]:
        cl = defaultdict(list)
        for (r, ds, ret, c, q), v in D.items():
            if r == rdr and c == cond and v["cites"] == 1:
                cl[(ds, q)].append(v["ex"])
        P.append(["bioasq+nfcorpus", rdr, cond, "extrapolation_given_cites", "level", *cluster_boot(cl)])
cl = defaultdict(list)
for (r, ds, ret, c, q), v in D.items():
    if v["cites"] == 1:
        cl[(ds, q)].append(v["ex"])
P.append(["bioasq+nfcorpus", "both readers", "all conditions", "extrapolation_given_cites", "level", *cluster_boot(cl)])
for name, dss in [("bioasq+nfcorpus", ("bioasq", "nfcorpus")), ("bioasq", ("bioasq",)), ("nfcorpus", ("nfcorpus",))]:
    for cond in ["base"] + CONDS:
        ks = np.array([v["k"] for (r, ds, ret, c, q), v in D.items() if r == "local" and c == cond and ds in dss])
        P.append([name, "(contexts)", cond, "top10_has_animal_only", "level", float(np.mean(ks >= 1)), None, None, len(ks)])
        P.append([name, "(contexts)", cond, "animal_only_per_top10", "level", float(np.mean(ks)), None, None, len(ks)])
wr("supporting/s5_answers_pooled.csv", ["datasets", "reader", "condition", "quantity", "kind", "estimate", "lo", "hi", "n"], P)
PP = {(r[0], r[1], r[2], r[3]): r for r in P}
copy_rounded("s5_effects.csv", "supporting/s5_effects_by_cell.csv")

rows = []
for rdr, rname in READERS:
    for name, dss in [("bioasq+nfcorpus", ("bioasq", "nfcorpus")), ("bioasq", ("bioasq",)), ("nfcorpus", ("nfcorpus",))]:
        for k in [0, 1, 2, 3]:
            g = [j for j in J if j["reader"] == rdr and j["cond"] == "base" and j["ds"] in dss and min(j["k"], 3) == k]
            n = len(g); c = sum(j["cites_nh"] for j in g); e = sum(j["ex"] for j in g); u = sum(j["unjudged"] for j in g)
            rows.append([rname, name, "3+" if k == 3 else k, n, c + u, c, e, u, 1 - (c + u) / n, (c - e) / n, e / n, u / n,
                         e / (c + u) if c + u else None, (e + u) / (c + u) if c + u else None, sum(j["any_nh"] for j in g) / n])
wr("fig4_answers_by_exposure.csv", ["reader", "datasets", "animal_only_in_top10", "answers", "n_rest_on_nonhuman_only", "n_judged", "n_not_stated", "n_unjudged",
                                    "share_none", "share_stated", "share_not_stated", "share_unjudged", "not_stated_rate_lo", "not_stated_rate_hi",
                                    "share_citing_nonhuman_in_any_bracket"], rows)
FIG4 = {(r[0], r[1], str(r[2])): r for r in rows}

rows = []
for rdr, rname in READERS:
    for name, dss in [("bioasq+nfcorpus", ("bioasq", "nfcorpus")), ("bioasq", ("bioasq",)), ("nfcorpus", ("nfcorpus",))]:
        for cond in ["base"] + CONDS + ["all conditions"]:
            g = [j for j in J if j["reader"] == rdr and j["ds"] in dss and (cond == "all conditions" or j["cond"] == cond)]
            n = len(g); c = sum(j["cites_nh"] for j in g); e = sum(j["ex"] for j in g); u = sum(j["unjudged"] for j in g)
            rows.append([rname, name, cond, n, c + u, c, e, u, e / n, (e + u) / n, e / (c + u) if c + u else None, (e + u) / (c + u) if c + u else None,
                         sum(j["any_nh"] for j in g) / n])
wr("supporting/s5_unjudged_bounds.csv", ["reader", "datasets", "condition", "answers", "n_rest_on_nonhuman_only", "n_judged", "n_not_stated", "n_unjudged",
                                         "extrap_rate_measured", "extrap_rate_if_all_unjudged_count", "not_stated_rate_lo", "not_stated_rate_hi",
                                         "share_citing_nonhuman_in_any_bracket"], rows)
BND = {(r[0], r[1], r[2]): r for r in rows}

qrels = {ds: json.load(open(f"{ROOT}/data/qq_{ds}.json", encoding="utf-8"))["qrels"] for ds in ["bioasq", "nfcorpus"]}
rows = []
for rdr, rname in READERS:
    for ds in ["bioasq", "nfcorpus"]:
        rel, nrel = [], []
        for j in J:
            if j["reader"] != rdr or j["cond"] != "base" or j["ds"] != ds or not j["cites_nh"]:
                continue
            a = A.get((j["ds"], j["retriever"], j["cond"], j["qid"], j["reader"]))
            if a is None:
                continue
            q = qrels[ds].get(j["qid"], {})
            nh = [a["docs"][c - 1] for c in j["cited"] if j["cats"][c - 1] in NONHUMAN]
            (rel if any(q.get(d, 0) > 0 for d in nh) else nrel).append(j["ex"])
        rows.append([rname, ds, "judged relevant", len(rel), mean(rel)])
        rows.append([rname, ds, "not judged relevant", len(nrel), mean(nrel)])
wr("supporting/s5_relevance_split.csv", ["reader", "ds", "cited_nonhuman_abstract", "answers", "not_stated_rate"], rows)
REL = {(r[0], r[1], r[2]): r for r in rows}
rows = []
for rdr, rname in READERS:
    b = [a for k, a in A.items() if k[4] == rdr and k[2] == "base"]
    allr = [j for j in J if j["reader"] == rdr]
    rows.append([rname, len([a for k, a in A.items() if k[4] == rdr]), len(allr), sum(j["cites_nh"] for j in allr), sum(j["ex"] for j in allr),
                 sum(j["unjudged"] for j in allr), len(b),
                 mean([len(a["full"]) for a in b]), mean([len(a["full"]) == 0 for a in b]), mean([a["any_nh"] for a in b]),
                 mean([a["sole"] or any(a["cats"][c - 1] in NONHUMAN for c in a["cited"]) for a in b]),
                 mean([len(a["cited"]) for a in b]), mean([len(a["cited"]) == 0 for a in b]),
                 mean([any(a["cats"][c - 1] in NONHUMAN for c in a["cited"]) for a in b]), mean([len(a["answer"].split()) for a in b])])
wr("supporting/s5_reader_stats.csv", ["reader", "answer_rows", "usable_answers", "judged", "not_stated", "unjudged", "base_answers",
                                      "base_cited_abstracts_per_answer", "base_share_no_citation", "base_share_citing_nonhuman_any_bracket",
                                      "base_share_resting_on_nonhuman_only", "single_bracket_parser_cited_per_answer", "single_bracket_parser_no_citation",
                                      "single_bracket_parser_cites_nonhuman", "base_mean_words"], rows)
RS = {r[0]: r for r in rows}

# ------------------------------------------------------------------ table 3: seven remedies
S4 = rd("s4_instructions.csv")


def s4(comp_, kind, col, ds="bioasq"):
    return [float(r[col]) for r in S4 if r["comparison"] == comp_ and r["kind"] == kind and r["ds"] == ds]


def rng_(v):
    return [min(v), max(v)] if v else [None, None]


def ans(cond):
    out = []
    for rdr in RD:
        r = PP[("bioasq+nfcorpus", rdr, cond, "extrapolation")]
        out += [r[5], r[6], r[7]]
    for rdr in RD:
        r = PP[("bioasq", rdr, cond, "correctness")]
        out += [r[5], r[6], r[7]]
    return out


NA12 = [None] * 12
hum_a = [float(r["shift"]) for r in cells("humans", "animal_only", ["bioasq"])]      # raw change vs no suffix, like the other rows
pat_a = [float(r["shift"]) for r in cells("patients", "animal_only", ["bioasq"])]
log("  in patients, animal-only on bioasq: raw mean", round(mean(pat_a), 4), "minus-neutral mean", round(SUMM[("patients", "animal_only", "bioasq")][5], 4),
    "base", round(SUMM[("patients", "animal_only", "bioasq")][4], 4))
mesh_n = [r[4] for r in FIX if r[0] == "bioasq" and r[2] == "mesh_demote"]
mesh_a = [r[7] for r in FIX if r[0] == "bioasq" and r[2] == "mesh_demote"]
T3 = [
    ["suffix in humans", "11 retrievers, vs no suffix (nDCG: mean of 11)", *rng_(hum_a), COST[("humans", "bioasq")][3], COST[("humans", "bioasq")][3], *NA12],
    ["suffix in patients", "11 retrievers, vs no suffix (nDCG: mean of 11)", *rng_(pat_a), COST[("patients", "bioasq")][3], COST[("patients", "bioasq")][3], *ans("patients")],
    ["embedder instruction: humans", "3 embedders, vs default instruction", *rng_(s4("instr_humans vs instr_default", "emb", "d_animal_only")),
     *rng_(s4("instr_humans vs instr_default", "emb", "d_ndcg10")), *NA12],
    ["reranker instruction: humans", "2 first stages, vs reranker default instruction", *rng_(s4("instr_humans vs instr_default", "rerank", "d_animal_only")),
     *rng_(s4("instr_humans vs instr_default", "rerank", "d_ndcg10")), *ans("rerank")],
    ["LLM pointwise scoring of the top 20", "Qwen3-Embedding first stage, vs LLM relevance-only scoring", *rng_(s4("llm_humans vs llm_relevant", "llm", "d_animal_only")),
     *rng_(s4("llm_humans vs llm_relevant", "llm", "d_ndcg10")), *NA12],
    ["label each abstract with its population", "context only, ranking unchanged", 0.0, 0.0, 0.0, 0.0, *ans("labelled")],
    ["MeSH demotion", "7 retrievers, vs no demotion", *rng_(mesh_a), *rng_(mesh_n), *ans("mesh_demote")],
]
wr("tab3_remedies.csv", ["remedy", "retrieval_basis_bioasq", "animal_only_top10_change_min", "animal_only_top10_change_max", "ndcg10_change_min", "ndcg10_change_max",
                         "extrap_change_gpt6luna", "extrap_gpt6luna_lo", "extrap_gpt6luna_hi", "extrap_change_qwen", "extrap_qwen_lo", "extrap_qwen_hi",
                         "bioasq_correct_change_gpt6luna", "correct_gpt6luna_lo", "correct_gpt6luna_hi",
                         "bioasq_correct_change_qwen", "correct_qwen_lo", "correct_qwen_hi"], T3)

# ------------------------------------------------------------------ figure 1: real cases
NONHUMAN_WORD = re.compile(r"\b(mice|mouse|murine|rats?|rodents?|animals?|fish|zebrafish|pigs?|porcine|dogs?|canine|rabbits?|monkeys?|primates?|sheep|"
                           r"bovine|chick(en)?s?|drosophila|yeast|in vitro|cell lines?|cultured|preclinical|models?)\b", re.I)
cand = []
for (ds, ret, cond, qid, rdr), a in A.items():
    if cond != "base" or rdr != "local":
        continue
    v = {r: D.get((r, ds, ret, "base", qid)) for r in RD}
    lab = {r: D.get((r, ds, ret, "labelled", qid)) for r in RD}
    if any(x is None for x in v.values()) or any(x is None for x in lab.values()):
        continue
    k = v["local"]["k"]
    if not 1 <= k <= 4 or sum(c == "human_only" for c in a["cats"]) < 3 or not all(x["ex"] == 1 for x in v.values()):
        continue
    if any(NONHUMAN_WORD.search(A[(ds, ret, "base", qid, r)]["answer"]) for r in RD):   # keep cases where no reader names a non-human population
        continue
    words = max(len(A[(ds, ret, "base", qid, r)]["answer"].split()) for r in RD)
    fixed = sum(x["ex"] == 0 for x in lab.values())
    cand.append((-fixed, -min(k, 3), len(a["question"].split()), words, ds, ret, qid))
cand.sort()
picked, seen = [], set()
for c in cand:
    if (c[4], c[6]) not in seen and c[2] <= 14:
        seen.add((c[4], c[6])); picked.append(c)
    if len(picked) == 8:
        break
need = {(c[4], c[5], cnd, c[6]) for c in picked for cnd in ("base",)}
ctx = {}
try:
    with open(f"{ROOT}/data/s5_contexts.jsonl", encoding="utf-8") as f:
        for l in f:
            if '"cond": "base"' not in l:
                continue
            c = json.loads(l)
            if (c["ds"], c["retriever"], c["cond"], c["qid"]) in need:
                ctx[(c["ds"], c["retriever"], c["qid"])] = c
except OSError:
    pass
with open(f"{O}/fig1_examples.md", "w", encoding="utf-8") as f:
    f.write("# Candidate cases for Figure 1\n\nReal queries from the base condition where both readers cite an animal-only or in-vitro abstract and the judge "
            "(gpt-6-luna, prompt v2) marks both answers as unqualified extrapolation. Kept only if the top 10 has at least 3 human-only and 1 to 4 animal-only "
            f"abstracts and neither answer contains a word for a non-human population. {len(cand)} query x retriever cases qualify; these are the first "
            f"{len(picked)} after sorting by: fixed by labelling, more animal-only abstracts in the top 10, shorter question. "
            "Answers are model output, quoted in full. Read a case before using it: the judge is wrong on roughly one answer in ten.\n")
    for i, (nf, nk, _, _, ds, ret, qid) in enumerate(picked):
        a0 = A[(ds, ret, "base", qid, "local")]
        f.write(f"\n## Case {i + 1}: {a0['question']}\n\n- Dataset {ds}, retriever {RNAME[ret]}, query id {qid}\n")
        f.write("- Top 10 by study population: " + ", ".join(f"{n + 1} {c}" for n, c in enumerate(a0["cats"])) + "\n")
        f.write(f"- With labelled abstracts, the unqualified extrapolation disappears for {-nf} of 2 readers\n")
        for rdr, rname in READERS:
            a = A[(ds, ret, "base", qid, rdr)]
            nh = [c for c in a["full"] if a["cats"][c - 1] in NONHUMAN]
            f.write(f"\n**{rname}** cites {a['full']}; animal-only or in-vitro among them: {nh}\n\n> " + " ".join(a["answer"].split()) + "\n")
        c = ctx.get((ds, ret, qid))
        nh_all = sorted({x for rdr in RD for x in A[(ds, ret, "base", qid, rdr)]["full"] if a0["cats"][x - 1] in NONHUMAN})
        f.write("\nCited animal-only or in-vitro abstracts:\n\n")
        for x in nh_all:
            t = ""
            if c:
                tt, ab = c["texts"][x - 1]
                t = " ".join(((tt or "") + " " + (ab or "")).split())[:220]
            f.write(f"- [{x}] PMID {a0['docs'][x - 1]} ({a0['cats'][x - 1]}): {t}...\n")
log(f"wrote fig1_examples.md: {len(picked)} cases of {len(cand)} candidates; contexts found {len(ctx)}")

# ------------------------------------------------------------------ check log: the numbers the guide quotes
log("\n== checks against the writing guide")
for s in ["humans", "patients", "rct", "mice"]:
    log(f"  human-only shift {PHRASE[s]:34s}", " / ".join(f"{FIG2[s]['mean_' + d]:+.3f}" for d in ["bioasq", "nfcorpus", "trec_covid"]),
        f"sig- {FIG2[s]['cells_sig_neg']} sig+ {FIG2[s]['cells_sig_pos']} logodds {FIG2[s]['logodds_bioasq']:+.2f} ndcg {FIG2[s]['ndcg10_change_bioasq']:+.3f}")
log("  fig2 row order:", [FIG2[s]["phrase"] for s in ORDER17])
log("  bioasq per retriever in humans:", " ".join(f"{c}={FIG2['humans'][c]:+.3f}{'*' if FIG2['humans'][c + '_sig'] else ''}" for _, c, _, _ in RETR))
for s in ["mice", "rats", "animals", "patients"]:
    log(f"  animal-only shift {PHRASE[s]:12s}", " / ".join(f"{SUMM[(s, 'animal_only', d)][5]:+.3f} (sig+ {SUMM[(s, 'animal_only', d)][8]}, sig- {SUMM[(s, 'animal_only', d)][9]})" for d in DSS),
        "base", " / ".join(f"{SUMM[(s, 'animal_only', d)][4]:.3f}" for d in DSS))
log("  in humans, implied both-share:", " / ".join(f"{BOTH[d][1]:+.3f} ({BOTH[d][4]}/{BOTH[d][5]} positive)" for d in DSS))
log("  sign agreement:", {k: f"{v[0]}/{v[1]}" for k, v in AGREE.items()})
log("  correlation medians:", dict(zip(["slope", "slope_lo", "slope_hi", "r2", "p", "r_all", "r_h10", "rho_h10", "r_h9", "rho_h9", "r_n7", "rho_n7"], [round(x, 2) for x in CORR["med"][2:14]])))
log("  n positive:", CORR["npos"][2:], "(last = cells where 'humans' is the lowest of the ten)")
log("  tab1:", {d: [fm(x, 3) for x in TAB1[d][1:]] for d in DSS})
for d in ["bioasq", "nfcorpus"]:
    e = [TAB2[(d, m)][7] for m in RKEYS if TAB2[(d, m)][7] is not None]
    log(f"  exposure {d}: {min(e):.3f} to {max(e):.3f} over {len(e)} retrievers; 3+ {min(TAB2[(d, m)][10] for m in RKEYS if TAB2[(d, m)][10] is not None):.3f}"
        f" to {max(TAB2[(d, m)][10] for m in RKEYS if TAB2[(d, m)][10] is not None):.3f}")
for a in ["adults", "children", "older_adults", "pregnancy"]:
    log(f"  axis {a:13s}", " ; ".join(f"{d} base {AX[(a, d)][3]:.3f} shift {AX[(a, d)][4]:+.3f} ratio {AX[(a, d)][9]:.2f} human-only {AX[(a, d)][10]:+.3f}" for d in ["bioasq", "nfcorpus"]))
for r in P:
    if r[0] == "bioasq+nfcorpus" and r[3] in ("extrapolation", "extrapolation_given_cites", "cites_nonhuman") or r[3] == "correctness":
        log("  ", r[0], r[1], r[2], r[3], r[4], f"{r[5]:+.3f}", f"[{fm(r[6], 3)}, {fm(r[7], 3)}]", "n", r[8])
for (rname, name, k), r in FIG4.items():
    if name == "bioasq+nfcorpus":
        log(f"  fig4 {rname:11s} k={k:2s} n {r[3]} rests {r[4]} judged {r[5]} not stated {r[6]} unjudged {r[7]} | shares none {r[8]:.3f} stated {r[9]:.3f} "
            f"not stated {r[10]:.3f} unjudged {r[11]:.3f} | rate {fm(r[12], 3)} to {fm(r[13], 3)} | any bracket {r[14]:.3f}")
for k_, r in BND.items():
    if k_[1] == "bioasq+nfcorpus":
        log(f"  bounds {k_[0]:11s} {k_[2]:14s} n {r[3]} rests {r[4]} judged {r[5]} not stated {r[6]} unjudged {r[7]} | rate {r[8]:.3f} to {r[9]:.3f} | "
            f"given {fm(r[10], 3)} to {fm(r[11], 3)} | any bracket {r[12]:.3f}")
log("  relevance split:", {k: (v[3], round(v[4], 3)) for k, v in REL.items()})
log("  reader stats:", RS)
for r in T3:
    log("  tab3", r[0], "|", " ".join(fm(x, 3) for x in r[2:]))
log("  s4 suffix rows (bioasq):", [(r["comparison"], r["retriever"], r["d_animal_only"][:7], r["d_ndcg10"][:7]) for r in S4 if r["kind"] == "suffix" and r["ds"] == "bioasq"])
log("  text demotion (bioasq):", [(r[1], fm(r[4], 3), fm(r[8], 2)) for r in FIX if r[0] == "bioasq" and r[2] == "text_demote"])
tot = sum(os.path.getsize(os.path.join(dp, fn)) for dp, _, fns in os.walk(O) for fn in fns)
log(f"\npaper_data size: {tot / 1024:.0f} KB in {sum(len(fns) for _, _, fns in os.walk(O))} files")
LOG.close()
