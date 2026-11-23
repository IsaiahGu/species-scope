"""P04 S7: when there is no MeSH - judging the study population from the text.

Test set: up to 20,000 hand-indexed documents (MEDLINE, no IndexingMethod attribute) of the three corpora; label =
the MeSH class (human_only, animal_only, both, vitro_only, neither). Three classifiers on title + abstract:
  regex   the population-word rule used in analyze.py
  bow     TF-IDF (1-2-grams) + multinomial logistic regression, trained on 200,000 MeSH-labelled PubMed abstracts
          from the baseline sample (data/pubmed_sample.jsonl.gz), disjoint from the test set
  llm     Qwen3.5-9B zero-shot (local vLLM), answer one of humans / animals / both / in_vitro / unclear
Per class precision and recall against MeSH; then each classifier's "animal-only" label is used for demotion and
compared with MeSH demotion (share of MeSH animal-only studies removed from the top 10, nDCG@10 cost); and the
exposure among documents that have no MeSH at all.
python s7_classify.py regex_bow | llm | apply   Output: results/s7_classifier.json, results/s7_doc_labels.parquet
"""

from pathlib import Path

WORKSPACE_ROOT = Path(__file__).resolve().parents[1].as_posix()
import gzip
import json
import random
import re
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, f"{WORKSPACE_ROOT}/shared")
sys.path.insert(0, f"{WORKSPACE_ROOT}/src")
from run_retrieval import load  # noqa: E402

ROOT = WORKSPACE_ROOT
VITRO = re.compile(r"Cell Line|Cells, Cultured|In Vitro Techniques|Cell Culture", re.I)
ANIMAL_TXT = re.compile(r"\b(mice|mouse|murine|rats?|rodents?|rabbits?|dogs?|pigs?|primates?|monkeys?|zebrafish|drosophila|hamsters?|sheep|cattle|animals?)\b", re.I)
VITRO_TXT = re.compile(r"\b(in vitro|cell lines?|cultured cells|cell culture)\b", re.I)
HUMAN_TXT = re.compile(r"\b(patients?|participants?|subjects?|volunteers?|women|men|adults?|children|humans?|people|cohort|trial)\b", re.I)
CLASSES = ["human_only", "animal_only", "both", "vitro_only", "neither"]
P = ("Title: {t}\n\nAbstract: {a}\n\nIn whom or what was this study done? Answer with one word: humans, animals, both "
     "(humans and animals), in_vitro (cells or tissue only), or unclear.")
MAP = {"humans": "human_only", "animals": "animal_only", "both": "both", "in_vitro": "vitro_only", "unclear": "neither"}


def mesh_class(terms):
    t = {m.split("/")[0].strip("* ") for m in terms}
    h, a = "Humans" in t, "Animals" in t
    return "both" if h and a else "human_only" if h else "animal_only" if a else ("vitro_only" if any(VITRO.search(x) for x in t) else "neither")


def regex_class(text):
    h, a, v = bool(HUMAN_TXT.search(text)), bool(ANIMAL_TXT.search(text)), bool(VITRO_TXT.search(text))
    return "both" if h and a else "human_only" if h else "animal_only" if a else ("vitro_only" if v else "neither")


def corpus_docs():
    idx = {j["pmid"]: j for j in (json.loads(l) for l in open(f"{ROOT}/data/pubmed_indexing.jsonl"))}
    docs = {}
    for ds in ["bioasq", "nfcorpus", "trec-covid"]:
        ids, titles, texts, qs, qrels = load(ds)
        d2p = json.load(open(f"{ROOT}/data/doc2pmid_{ds}.json"))
        for d, t, x in zip(ids, titles, texts):
            p = d2p.get(d)
            if p:
                docs[p] = {"pmid": p, "title": t or "", "abstract": x or "", "idx": idx.get(p)}
    return docs


def regex_bow():
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import classification_report
    docs = corpus_docs()
    manual = [d for d in docs.values() if d["idx"] and d["idx"]["indexing"] == "Manual" and d["idx"]["status"] == "MEDLINE"
              and d["idx"]["mesh"] and len(d["abstract"].split()) >= 30]
    random.Random(0).shuffle(manual)
    test = manual[:20000]
    tp = {d["pmid"] for d in test}
    tr = []
    for l in gzip.open(f"{ROOT}/data/pubmed_sample.jsonl.gz", "rt"):
        j = json.loads(l)
        if j["pmid"] not in tp and j.get("indexing") == "Manual":
            tr.append(j)
        if len(tr) >= 200000:
            break
    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=3, max_features=300000, sublinear_tf=True, stop_words="english")
    Xtr = vec.fit_transform([f"{j['title']} {j['abstract']}" for j in tr])
    ytr = [mesh_class(j["mesh"]) for j in tr]
    clf = LogisticRegression(max_iter=300, C=4.0, n_jobs=4).fit(Xtr, ytr)
    texts = [f"{d['title']} {d['abstract']}" for d in test]
    y = [mesh_class(d["idx"]["mesh"]) for d in test]
    pred_bow = list(clf.predict(vec.transform(texts)))
    pred_re = [regex_class(t) for t in texts]
    out = {"n_test": len(test), "n_train": len(tr), "train_class_share": pd.Series(ytr).value_counts(normalize=True).round(4).to_dict(),
           "test_class_share": pd.Series(y).value_counts(normalize=True).round(4).to_dict(),
           "regex": classification_report(y, pred_re, labels=CLASSES, output_dict=True, zero_division=0),
           "bow": classification_report(y, pred_bow, labels=CLASSES, output_dict=True, zero_division=0)}
    json.dump(out, open(f"{ROOT}/results/s7_classifier.json", "w"), indent=1)
    # labels for every corpus document (MeSH or not), for demotion and S6
    allp = sorted(docs)
    alltext = [f"{docs[p]['title']} {docs[p]['abstract']}" for p in allp]
    lab = pd.DataFrame({"pmid": allp, "regex": [regex_class(t) for t in alltext], "bow": clf.predict(vec.transform(alltext)),
                        "mesh": [mesh_class(docs[p]["idx"]["mesh"]) if docs[p]["idx"] and docs[p]["idx"]["mesh"] else None for p in allp],
                        "test": [p in tp for p in allp]})
    lab["text_label"] = lab["bow"]
    lab.to_parquet(f"{ROOT}/results/s7_doc_labels.parquet")
    for m in ("regex", "bow"):
        r = out[m]
        print(m, {c: (round(r[c]["precision"], 3), round(r[c]["recall"], 3)) for c in CLASSES})


def llm():
    from llm import LLM
    from sklearn.metrics import classification_report
    lab = pd.read_parquet(f"{ROOT}/results/s7_doc_labels.parquet")
    docs = corpus_docs()
    lm = LLM("local", cache_name="p04_s7_llm")
    ps = lab.pmid.tolist()
    outs = lm.chat_many([[{"role": "user", "content": P.format(t=docs[p]["title"], a=docs[p]["abstract"][:2500])}] for p in ps], max_tokens=4)
    lab["llm"] = [MAP.get(re.sub(r"[^a-z_]", "", (o[0]["text"] or "").strip().lower().split()[0]) if (o[0]["text"] or "").strip() else "", "neither") for o in outs]
    lab.to_parquet(f"{ROOT}/results/s7_doc_labels.parquet")
    t = lab[lab.test]
    out = json.load(open(f"{ROOT}/results/s7_classifier.json"))
    out["llm"] = classification_report(t.mesh, t.llm, labels=CLASSES, output_dict=True, zero_division=0)
    json.dump(out, open(f"{ROOT}/results/s7_classifier.json", "w"), indent=1)
    print("llm", {c: (round(out["llm"][c]["precision"], 3), round(out["llm"][c]["recall"], 3)) for c in CLASSES})


def apply():
    from metrics import ndcg_at_k
    lab = pd.read_parquet(f"{ROOT}/results/s7_doc_labels.parquet").set_index("pmid")
    out = json.load(open(f"{ROOT}/results/s7_classifier.json"))
    rows = []
    for ds in ["bioasq", "nfcorpus", "trec-covid"]:
        qq = json.load(open(f"{ROOT}/data/qq_{ds}.json"))
        d2p = json.load(open(f"{ROOT}/data/doc2pmid_{ds}.json"))
        R = pd.read_parquet(f"{ROOT}/results/runs_sweep_{ds}.parquet")
        R = R[R.scope == "base"]
        for (m, q), g in R.groupby(["retriever", "qid"]):
            docs = g.sort_values("rank").doc.tolist()
            pm = [d2p.get(d) for d in docs]
            rel = qq["qrels"].get(q, {})
            mesh = [lab.mesh.get(p) if p in lab.index else None for p in pm]
            for how in ("none", "mesh", "regex", "bow", "llm"):
                if how != "none" and how != "mesh" and how not in lab.columns:
                    continue
                flag = [False] * len(docs) if how == "none" else [(mesh[i] == "animal_only") if how == "mesh" else
                                                                     (pm[i] in lab.index and lab.loc[pm[i], how] == "animal_only") for i in range(len(docs))]
                order = [d for d, f in zip(docs, flag) if not f] + [d for d, f in zip(docs, flag) if f]
                top = order[:10]
                mt = [lab.mesh.get(d2p.get(d)) if d2p.get(d) in lab.index else None for d in top]
                no_mesh = [d for d in top if not (d2p.get(d) in lab.index and lab.mesh.get(d2p.get(d)))]
                rows.append({"ds": ds, "retriever": m, "qid": q, "demote": how, "ndcg10": ndcg_at_k(order, rel, 10),
                             "mesh_animal10": float(np.mean([x == "animal_only" for x in mt])),
                             "n_no_mesh10": len(no_mesh),
                             "pred_animal_no_mesh10": sum(1 for d in no_mesh if d2p.get(d) in lab.index and
                                                          lab.loc[d2p.get(d), "bow"] == "animal_only")})
    D = pd.DataFrame(rows)
    S = D.groupby(["ds", "retriever", "demote"])[["ndcg10", "mesh_animal10", "n_no_mesh10", "pred_animal_no_mesh10"]].mean().reset_index()
    out["demotion"] = S.to_dict(orient="records")
    json.dump(out, open(f"{ROOT}/results/s7_classifier.json", "w"), indent=1, default=float)
    print(S.round(4).to_string(index=False))


if __name__ == "__main__":
    {"regex_bow": regex_bow, "llm": llm, "apply": apply}[sys.argv[1]]()
