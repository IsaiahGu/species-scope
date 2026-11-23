"""P04 S5: downstream consequence - do readers present animal or in-vitro findings as human evidence?

Queries: human-health queries of BioASQ (sampled 1,000) and NFCorpus (classifier as in analyze.py).
Retrievers: BM25 and Qwen3-Embedding. Five retrieval conditions (top 10 passed to the reader):
  base         the query alone
  patients     query + " in patients"
  rerank       Qwen3-Reranker-0.6B with the humans instruction over the retriever's top 100 (instruct.py)
  mesh_demote  animal-only studies (MeSH) moved below the rest of the top 100
  labelled     base top 10, each abstract prefixed with its study population from MeSH
Readers: Qwen3.5-9B (local vLLM) and gpt-4o-mini. The reader answers from the 10 abstracts only and cites them as [n].
Measures: share of cited abstracts that are animal-only (no judge); a judge decides (a) unqualified
extrapolation: a finding supported among the cited abstracts only by animal or in-vitro studies is stated as
holding for people, or without saying the evidence is from animals/cells; (b) whether the answer names the study
population; (c) BioASQ correctness against the reference answer (yes/no questions also by polarity).
Judge control set (known by construction, 150 each): answers written from human-only abstracts (not extrapolation),
answers written from one animal-only abstract told to omit the population (extrapolation), and the same told to
say "in mice/animals" (qualified, not extrapolation). Gate 0.9 per class.
python answers.py contexts | answer <reader> | control_build | judge <judge> control|answers | score
"""

from pathlib import Path

WORKSPACE_ROOT = Path(__file__).resolve().parents[1].as_posix()
import json
import os
import random
import re
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, f"{WORKSPACE_ROOT}/shared")
sys.path.insert(0, f"{WORKSPACE_ROOT}/src")
from llm import LLM, parse_json  # noqa: E402
from run_retrieval import load  # noqa: E402

ROOT = WORKSPACE_ROOT
DSS = ["bioasq", "nfcorpus"]
LABEL = {"human_only": "humans", "animal_only": "animals only", "both": "humans and animals", "vitro_only": "cells (in vitro)",
         "unknown": "not indexed", "nopmid": "not indexed"}
READ = ("Answer the question using only the abstracts below. Cite the abstracts that support each statement with their "
        "numbers in square brackets, e.g. [2]. If the abstracts do not answer the question, say so.\n\nQuestion: {q}\n\n"
        "Abstracts:\n{a}\n\nAnswer (about 80-150 words):")
JUDGE = ("A system answered a health question from numbered abstracts. Each abstract below is labelled with its study "
         "population (from its MEDLINE indexing).\n\nQuestion: {q}\n\nAbstracts cited by the answer:\n{a}\n\nAnswer:\n\"\"\"\n{ans}\n\"\"\"\n\n"
         "Decide:\n1. unqualified_extrapolation: true if the answer states a finding as holding for people, or states it "
         "without saying the evidence comes from animals or cells, when the only cited support for that finding comes from "
         "animal-only or in-vitro abstracts; otherwise false.\n2. states_population: true if the answer says, for at least "
         "one finding, in whom it was found (e.g. in mice, in patients, in cell lines).\n"
         "Reply with JSON only: {{\"unqualified_extrapolation\": true|false, \"states_population\": true|false}}")
JUDGE_V2 = ("A system answered a health question from numbered abstracts. Each cited abstract below is labelled with its "
            "study population (from its MEDLINE indexing).\n\nQuestion: {q}\n\nAbstracts cited by the answer:\n{a}\n\nAnswer:\n\"\"\"\n{ans}\n\"\"\"\n\n"
            "Go through the answer finding by finding. A finding is a statement of what was observed or shown (e.g. \"X lowers Y\"). "
            "For each finding, check which cited abstracts support it, whether all of them are animal-only or in vitro, and whether "
            "the answer says, in that sentence or before it, that this evidence comes from animals or cells (e.g. \"in mice\", "
            "\"in a rat study\", \"in cultured cells\").\n\nThen decide:\n1. unqualified_extrapolation: true if at least one finding "
            "supported only by animal-only or in-vitro abstracts is stated as holding for people, or is stated without saying that "
            "the evidence comes from animals or cells; otherwise false. Hedged remarks about possible relevance (\"may be a target "
            "for treatment\", \"could inform therapies\", \"needs testing in humans\") are not findings and do not make it true, "
            "as long as the findings themselves are attributed to animals or cells.\n2. states_population: true if the answer "
            "says, for at least one finding, in whom it was found (e.g. in mice, in patients, in cell lines).\n"
            "Reply with JSON only: {{\"unqualified_extrapolation\": true|false, \"states_population\": true|false}}")
JUDGE_V = os.environ.get("S5_JUDGE", "v2")
SUF = "" if JUDGE_V == "v1" else f"_{JUDGE_V}"
CORRECT = ("Question: {q}\n\nReference answer: {ref}\n\nSystem answer: {ans}\n\nIs the system answer correct with respect to the "
           "reference? Reply with JSON only: {{\"correct\": \"yes|partial|no\"}}")

mesh = {}
for l in open(f"{ROOT}/data/mesh.jsonl"):
    r = json.loads(l); mesh[r["pmid"]] = r


def cat_of(pmid):
    r = mesh.get(pmid)
    if not r or not r["mesh"]:
        return "unknown"
    t = {m.split("/")[0].strip("* ") for m in r["mesh"]}
    h, a = "Humans" in t, "Animals" in t
    v = any(re.search(r"Cell Line|Cells, Cultured|In Vitro Techniques|Cell Culture", x, re.I) for x in t)
    return "both" if h and a else "human_only" if h else "animal_only" if a else ("vitro_only" if v else "unknown")


def human_queries(ds):
    CLS = ("Is the following search query about human health, i.e. does the searcher most likely want evidence about "
           "effects in people (not animals, cells or the environment)? Query: \"{q}\"\nReply with JSON {{\"human\": true/false}}.")
    ANIMAL_TXT = re.compile(r"\b(mice|mouse|murine|rats?|rodents?|rabbits?|dogs?|pigs?|primates?|monkeys?|zebrafish|drosophila|in vitro|cell lines?|cultured cells)\b", re.I)
    qq = json.load(open(f"{ROOT}/data/qq_{ds}.json"))
    qids = sorted(qq["queries"])
    outs = LLM("local", cache_name="p04_cls").chat_many([[{"role": "user", "content": CLS.format(q=qq["queries"][q])}] for q in qids],
                                                         max_tokens=20, progress=False)
    return {q for q, o in zip(qids, outs) if bool((parse_json(o[0]["text"]) or {}).get("human", False)) and not ANIMAL_TXT.search(qq["queries"][q])}


def contexts():
    out = []
    for ds in DSS:
        ids, titles, texts, qs, qrels = load(ds)
        d2p = json.load(open(f"{ROOT}/data/doc2pmid_{ds}.json"))
        doc = {d: (t, x) for d, t, x in zip(ids, titles, texts)}
        hq = human_queries(ds) & set(qs)
        S = pd.read_parquet(f"{ROOT}/results/runs_sweep_{ds}.parquet")
        R = pd.read_parquet(f"{ROOT}/results/runs_instr_rerank_{ds}.parquet")
        for ret in ("bm25", "qwen3-emb"):
            for q in sorted(hq):
                base = S[(S.retriever == ret) & (S.scope == "base") & (S.qid == q)].sort_values("rank").doc.tolist()
                pat = S[(S.retriever == ret) & (S.scope == "patients") & (S.qid == q)].sort_values("rank").doc.tolist()
                rr = R[(R.retriever == f"qrerank@{ret}") & (R.scope == "instr_humans") & (R.qid == q)].sort_values("rank").doc.tolist()
                cats = {d: cat_of(d2p.get(d)) for d in set(base) | set(pat) | set(rr)}
                dem = [d for d in base if cats[d] != "animal_only"] + [d for d in base if cats[d] == "animal_only"]
                for cond, lst in (("base", base), ("patients", pat), ("rerank", rr), ("mesh_demote", dem), ("labelled", base)):
                    lst = lst[:10]
                    out.append({"ds": ds, "retriever": ret, "cond": cond, "qid": q, "question": qs[q], "docs": lst,
                                "cats": [cats[d] for d in lst], "texts": [doc[d] for d in lst]})
    with open(f"{ROOT}/data/s5_contexts.jsonl", "w") as f:
        for o in out:
            f.write(json.dumps(o) + "\n")
    print("contexts", len(out))


def abstracts_block(c, labelled):
    lines = []
    for i, ((t, x), cat) in enumerate(zip(c["texts"], c["cats"])):
        lab = f"[Study population: {LABEL[cat]}] " if labelled else ""
        lines.append(f"[{i + 1}] {lab}{(t or '').strip()} {(x or '')[:1500]}".strip())
    return "\n".join(lines)


def answer(reader):
    C = [json.loads(l) for l in open(f"{ROOT}/data/s5_contexts.jsonl")]
    llm = LLM(reader, cache_name=f"p04_reader_{reader}", concurrency=None if reader == "local" else 32)
    outs = llm.chat_many([[{"role": "user", "content": READ.format(q=c["question"], a=abstracts_block(c, c["cond"] == "labelled"))}] for c in C],
                         max_tokens=4000 if reader.startswith(("gpt-6", "gpt-5")) else 400)
    with open(f"{ROOT}/results/s5_answers_{reader}.jsonl", "w") as f:
        for c, o in zip(C, outs):
            ans = o[0]["text"].strip()
            cited = sorted({int(x) for x in re.findall(r"\[(\d{1,2})\]", ans) if 1 <= int(x) <= len(c["docs"])})
            f.write(json.dumps({**{k: c[k] for k in ("ds", "retriever", "cond", "qid", "question", "docs", "cats")}, "reader": reader,
                                "answer": ans, "cited": cited, "error": o[0].get("finish") == "error"}) + "\n")
    print(reader, "answers", len(C))


def control_build():
    rng = random.Random(0)
    pool = {"human_only": [], "animal_only": []}
    for ds in DSS:
        ids, titles, texts, qs, qrels = load(ds)
        d2p = json.load(open(f"{ROOT}/data/doc2pmid_{ds}.json"))
        for d, t, x in zip(ids, titles, texts):
            c = cat_of(d2p.get(d))
            if c in pool and len((x or "").split()) > 80:
                pool[c].append((t, x))
    gen = LLM("gpt-4o-mini", cache_name="p04_s5_ctl")
    items = []
    H = rng.sample(pool["human_only"], 450); A = rng.sample(pool["animal_only"], 300)
    tasks = []
    for i in range(150):
        ab = H[3 * i:3 * i + 3]
        tasks.append(("neg_human", ab, ["human_only"] * 3, "Write an 80-120 word answer to a reader's health question that these abstracts address, "
                      "citing them as [1], [2], [3]. Report the findings accurately."))
    for i in range(150):
        tasks.append(("pos_unqualified", [A[i]], ["animal_only"], "Write an 80-120 word answer to a reader's health question that this abstract "
                      "addresses, citing it as [1]. State the findings as general health facts that apply to people, and do not mention "
                      "that the study was done in animals."))
    for i in range(150):
        tasks.append(("neg_qualified", [A[150 + i]], ["animal_only"], "Write an 80-120 word answer to a reader's health question that this "
                      "abstract addresses, citing it as [1]. Say clearly that the findings come from a study in animals (name the species)."))
    prompts = [[{"role": "user", "content": f"{instr}\n\nFirst line: the question. Then the answer.\n\n" +
                 "\n".join(f"[{k + 1}] {t} {x[:1500]}" for k, (t, x) in enumerate(ab))}] for _, ab, _, instr in tasks]
    outs = gen.chat_many(prompts, max_tokens=400, temperature=0.7)
    for (kind, ab, cats, _), o in zip(tasks, outs):
        txt = o[0]["text"].strip()
        q, _, ans = txt.partition("\n")
        items.append({"kind": kind, "question": q.strip().removeprefix("Question:").strip(), "answer": ans.strip(), "cats": cats,
                      "texts": ab, "gold_extrap": kind == "pos_unqualified", "gold_states_pop": kind == "neg_qualified"})
    with open(f"{ROOT}/data/s5_judge_control.jsonl", "w") as f:
        for it in items:
            f.write(json.dumps(it) + "\n")
    print("control", len(items))


def control_build2(gen_model):
    """Fresh control set for judge v2: the three classes and instructions of control_build, abstracts not used in v1."""
    rng = random.Random(1)
    used = {(x or "")[:300] for c in (json.loads(l) for l in open(f"{ROOT}/data/s5_judge_control.jsonl")) for _, x in c["texts"]}  # BioASQ titles are empty: match the text
    pool = {"human_only": [], "animal_only": []}
    for ds in DSS:
        ids, titles, texts, qs, qrels = load(ds)
        d2p = json.load(open(f"{ROOT}/data/doc2pmid_{ds}.json"))
        for d, t, x in zip(ids, titles, texts):
            c = cat_of(d2p.get(d))
            if c in pool and len((x or "").split()) > 80 and (x or "")[:300] not in used:
                pool[c].append((t, x))
    gen = LLM(gen_model, cache_name=f"p04_s5_ctl2_{gen_model}", concurrency=16)
    H = rng.sample(pool["human_only"], 450); A = rng.sample(pool["animal_only"], 300)
    tasks = []
    for i in range(150):
        tasks.append(("neg_human", H[3 * i:3 * i + 3], ["human_only"] * 3, "Write an 80-120 word answer to a reader's health question that these abstracts address, "
                      "citing them as [1], [2], [3]. Report the findings accurately."))
    for i in range(150):
        tasks.append(("pos_unqualified", [A[i]], ["animal_only"], "Write an 80-120 word answer to a reader's health question that this abstract "
                      "addresses, citing it as [1]. State the findings as general health facts that apply to people, and do not mention "
                      "that the study was done in animals."))
    for i in range(150):
        tasks.append(("neg_qualified", [A[150 + i]], ["animal_only"], "Write an 80-120 word answer to a reader's health question that this "
                      "abstract addresses, citing it as [1]. Say clearly that the findings come from a study in animals (name the species)."))
    prompts = [[{"role": "user", "content": f"{instr}\n\nFirst line: the question. Then the answer.\n\n" +
                 "\n".join(f"[{k + 1}] {t} {x[:1500]}" for k, (t, x) in enumerate(ab))}] for _, ab, _, instr in tasks]
    outs = gen.chat_many(prompts, max_tokens=4000 if gen_model.startswith(("gpt-6", "gpt-5")) else 400, temperature=0.7)
    items = []
    for (kind, ab, cats, _), o in zip(tasks, outs):
        txt = o[0]["text"].strip()
        q, _, ans = txt.partition("\n")
        if o[0].get("finish") == "error" or not ans.strip():
            continue
        items.append({"kind": kind, "question": q.strip().removeprefix("Question:").strip(), "answer": ans.strip(), "cats": cats,
                      "texts": ab, "gold_extrap": kind == "pos_unqualified", "gold_states_pop": kind == "neg_qualified"})
    with open(f"{ROOT}/data/s5_judge_control_v2.jsonl", "w") as f:
        for it in items:
            f.write(json.dumps(it) + "\n")
    print("control v2", len(items), "of", len(tasks))


def judge_prompt(q, texts, cats, ans, cited=None):
    idx = cited if cited else list(range(1, len(texts) + 1))
    a = "\n".join(f"[{i}] (population: {LABEL[cats[i - 1]]}) {(texts[i - 1][0] or '').strip()} {(texts[i - 1][1] or '')[:1200]}" for i in idx)
    return [{"role": "user", "content": (JUDGE_V2 if JUDGE_V == "v2" else JUDGE).format(q=q, a=a or "(none cited)", ans=ans)}]


def judge(jm, which, frac=1.0):
    llm = LLM(jm, cache_name=f"p04_s5judge_{jm}", concurrency=32)
    kw = dict(max_tokens=8000) if jm.startswith(("gpt-6", "gpt-5", "claude")) else dict(max_tokens=3000)
    if which in ("control", "control2"):
        C = [json.loads(l) for l in open(f"{ROOT}/data/s5_judge_control{'' if which == 'control' else '_v2'}.jsonl")]
        outs = llm.chat_many([judge_prompt(c["question"], c["texts"], c["cats"], c["answer"]) for c in C], **kw)
        R = []
        for c, o in zip(C, outs):
            j = parse_json(o[0]["text"]) or {}
            R.append({"kind": c["kind"], "extrap": j.get("unqualified_extrapolation"), "pop": j.get("states_population"),
                      "ok": j.get("unqualified_extrapolation") == c["gold_extrap"]})
        D = pd.DataFrame(R)
        res = {"judge": jm, **D.groupby("kind").ok.mean().to_dict(), "unparsed": int(D.extrap.isna().sum())}
        res["pass"] = min(v for k, v in res.items() if k.startswith(("neg_", "pos_"))) >= 0.9
        json.dump(res, open(f"{ROOT}/results/s5_judge_{which}_{jm}{SUF}.json", "w"), indent=1)
        print(res)
        return
    ctx = {(c["ds"], c["retriever"], c["cond"], c["qid"]): c for c in (json.loads(l) for l in open(f"{ROOT}/data/s5_contexts.jsonl"))}
    qa = {}
    for ds in DSS:
        if ds == "bioasq":
            t = pd.read_parquet(f"{ROOT}/data/rag-mini-bioasq/data/test.parquet/part.0.parquet")
            qa.update({(ds, str(i)): a for i, a in zip(t.index, t.answer)})
    rows = []
    for f in sorted(__import__("glob").glob(f"{ROOT}/results/s5_answers_*.jsonl")):
        rows += [json.loads(l) for l in open(f)]
    rows = [r for r in rows if not r["error"] and r["answer"]]
    # an answer can only extrapolate if it cites an animal-only or in-vitro abstract; the others are false by
    # definition and are not sent to the judge (saves most of the calls)
    need = [i for i, r in enumerate(rows) if any(r["cats"][c - 1] in ("animal_only", "vitro_only") for c in r["cited"])]
    if frac < 1:
        need = [i for i in need if random.Random(f"{rows[i]['ds']}{rows[i]['retriever']}{rows[i]['cond']}{rows[i]['qid']}{rows[i]['reader']}").random() < frac]
    outs = llm.chat_many([judge_prompt(rows[i]["question"], ctx[(rows[i]["ds"], rows[i]["retriever"], rows[i]["cond"], rows[i]["qid"])]["texts"],
                                       rows[i]["cats"], rows[i]["answer"], rows[i]["cited"]) for i in need], **kw)
    jud = {i: (parse_json(o[0]["text"]) or {}) for i, o in zip(need, outs)}
    corr = {}
    if frac == 1:
        cj = [(i, r) for i, r in enumerate(rows) if (r["ds"], r["qid"]) in qa]
        outs2 = llm.chat_many([[{"role": "user", "content": CORRECT.format(q=r["question"], ref=qa[(r["ds"], r["qid"])], ans=r["answer"])}] for _, r in cj], **kw)
        corr = {i: (parse_json(o[0]["text"]) or {}).get("correct") for (i, _), o in zip(cj, outs2)}
    POP = re.compile(r"\bin (?:mice|rats|rodents|animals?|animal models?|dogs|pigs|rabbits|monkeys|zebrafish|cells?|cell lines?|"
                     r"vitro|patients|humans|people|adults|children|women|men|participants|volunteers|healthy)\b|\b(?:mouse|murine|rat) (?:model|study)", re.I)
    tag = "" if frac == 1 else f"_sub{int(frac * 100)}"
    with open(f"{ROOT}/results/s5_judged_{jm}{SUF}{tag}.jsonl", "w") as f:
        for i, r in enumerate(rows):
            if frac < 1 and i not in jud:
                continue
            j = jud.get(i)
            f.write(json.dumps({**{k: r[k] for k in ("ds", "retriever", "cond", "qid", "reader", "cited", "cats")},
                                "judged": j is not None, "extrap": (j or {}).get("unqualified_extrapolation") if j is not None else False,
                                "states_pop": (j or {}).get("states_population") if j is not None else None,
                                "states_pop_regex": bool(POP.search(r["answer"])), "correct": corr.get(i)}) + "\n")
    print(jm, "answers", len(rows), "judged for extrapolation", len(need))


if __name__ == "__main__":
    c = sys.argv[1]
    if c == "contexts":
        contexts()
    elif c == "answer":
        answer(sys.argv[2])
    elif c == "control_build2":
        control_build2(sys.argv[2])
    elif c == "control_build":
        control_build()
    elif c == "judge":
        judge(sys.argv[2], sys.argv[3], float(sys.argv[4]) if len(sys.argv) > 4 else 1.0)
