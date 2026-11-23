"""Shared ranking metrics and bootstrap helpers."""
from __future__ import annotations

import numpy as np


def rbo(S, T, p=0.9):
    """Extrapolated rank-biased overlap (Webber et al. 2010) for two equal-depth rankings."""
    k = min(len(S), len(T))
    if k == 0:
        return 0.0
    S, T = list(S[:k]), list(T[:k])
    seen_s, seen_t, x, acc = set(), set(), 0, 0.0
    for d in range(1, k + 1):
        a, b = S[d - 1], T[d - 1]
        if a == b:
            x += 1
        else:
            x += (a in seen_t) + (b in seen_s)
        seen_s.add(a); seen_t.add(b)
        acc += (x / d) * p ** d
    return float((x / k) * p ** k + (1 - p) / p * acc)


def ndcg_at_k(ranked, rel: dict, k=10):
    g = [rel.get(d, 0) for d in ranked[:k]]
    dcg = sum((2 ** r - 1) / np.log2(i + 2) for i, r in enumerate(g))
    ideal = sorted(rel.values(), reverse=True)[:k]
    idcg = sum((2 ** r - 1) / np.log2(i + 2) for i, r in enumerate(ideal))
    return dcg / idcg if idcg > 0 else np.nan


def average_precision(ranked, rel: dict):
    nrel = sum(1 for v in rel.values() if v > 0)
    if nrel == 0:
        return np.nan
    hits, s = 0, 0.0
    for i, d in enumerate(ranked):
        if rel.get(d, 0) > 0:
            hits += 1
            s += hits / (i + 1)
    return s / nrel


def recall_at_k(ranked, rel: dict, k=10):
    pos = {d for d, v in rel.items() if v > 0}
    if not pos:
        return np.nan
    return len(pos & set(ranked[:k])) / len(pos)


def bootstrap_ci(x, B=2000, seed=0, stat=np.nanmean):
    x = np.asarray(x, float)
    x = x[~np.isnan(x)]
    if len(x) == 0:
        return (np.nan, np.nan, np.nan)
    rng = np.random.default_rng(seed)
    bs = [stat(x[rng.integers(0, len(x), len(x))]) for _ in range(B)]
    return (float(stat(x)), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5)))


def cluster_bootstrap(df, cluster, fn, B=1000, seed=0):
    """Resample clusters with replacement; fn(df) -> scalar."""
    rng = np.random.default_rng(seed)
    groups = {k: g for k, g in df.groupby(cluster)}
    keys = list(groups)
    est = fn(df)
    import pandas as pd
    vals = []
    for _ in range(B):
        pick = rng.integers(0, len(keys), len(keys))
        vals.append(fn(pd.concat([groups[keys[i]] for i in pick], ignore_index=True)))
    return float(est), float(np.nanpercentile(vals, 2.5)), float(np.nanpercentile(vals, 97.5))
