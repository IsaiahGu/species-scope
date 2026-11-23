"""P04 step 1: PMIDs for every document, then MeSH headings and publication types from PubMed.

BioASQ-mini: passage ids are PMIDs.  TREC-COVID: cord_uid -> pubmed_id (CORD-19 2020-07-16).
NFCorpus: BEIR ids (MED-x) have no PMID, so each title is searched in PubMed ([ti], exact
single hit only).  E-utilities without an API key: <= 3 requests/s.
Outputs data/doc2pmid_<ds>.json, data/mesh.jsonl
"""

from pathlib import Path

WORKSPACE_ROOT = Path(__file__).resolve().parents[1].as_posix()
import json
import os
import re
import time
import xml.etree.ElementTree as ET

import pandas as pd
import requests

ROOT = WORKSPACE_ROOT
B = f"{WORKSPACE_ROOT}/data/beir"
E = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
S = requests.Session()
S.headers["User-Agent"] = "species-scope-research/0.1"
BASE = {"tool": "species_scope_research"}
_last = [0.0]


def call(endpoint, data):
    for attempt in range(8):
        wait = 0.36 - (time.time() - _last[0])
        if wait > 0:
            time.sleep(wait)
        _last[0] = time.time()
        try:
            r = S.post(f"{E}/{endpoint}", data={**BASE, **data}, timeout=120)
            if r.status_code == 200:
                return r.text
            time.sleep(3 * (attempt + 1))
        except Exception:  # noqa: BLE001
            time.sleep(3 * (attempt + 1))
    return None


# ---- doc -> pmid maps
maps = {}
p = pd.read_parquet(f"{ROOT}/data/rag-mini-bioasq/data/passages.parquet/part.0.parquet")
maps["bioasq"] = {str(i): str(i) for i in p.index}
m = pd.read_csv(f"{ROOT}/data/cord19_metadata_2020-07-16.csv", usecols=["cord_uid", "pubmed_id"], dtype=str).dropna()
tc = pd.read_parquet(f"{B}/trec-covid/corpus")
uid2pm = dict(zip(m.cord_uid, m.pubmed_id))
maps["trec-covid"] = {d: uid2pm[d].split(".")[0] for d in tc["_id"].astype(str) if d in uid2pm}
nf_path = f"{ROOT}/data/doc2pmid_nfcorpus.json"
if os.path.exists(nf_path):
    maps["nfcorpus"] = json.load(open(nf_path))
else:
    nf = pd.read_parquet(f"{B}/nfcorpus/corpus")
    out = {}
    for i, (did, title) in enumerate(zip(nf["_id"].astype(str), nf["title"].fillna(""))):
        t = re.sub(r"[\[\]\(\)\"]", " ", title).strip().rstrip(".")
        if len(t) < 10:
            continue
        txt = call("esearch.fcgi", {"db": "pubmed", "term": f"{t}[ti]", "retmode": "json", "retmax": "3"})
        try:
            ids = json.loads(txt)["esearchresult"]["idlist"] if txt else []
        except Exception:  # noqa: BLE001
            ids = []
        if len(ids) == 1:
            out[did] = ids[0]
        if i % 500 == 0:
            print("nfcorpus esearch", i, len(out), flush=True)
    json.dump(out, open(nf_path, "w"))
    maps["nfcorpus"] = out
for k, v in maps.items():
    json.dump(v, open(f"{ROOT}/data/doc2pmid_{k}.json", "w"))
    print(k, "docs with pmid", len(v), flush=True)

# ---- MeSH via efetch (MEDLINE text format)
pmids = sorted({x for v in maps.values() for x in v.values()})
done = set()
outp = f"{ROOT}/data/mesh.jsonl"
if os.path.exists(outp):
    done = {json.loads(l)["pmid"] for l in open(outp)}
todo = [x for x in pmids if x not in done]
print("pmids", len(pmids), "todo", len(todo), flush=True)
with open(outp, "a") as f:
    for s in range(0, len(todo), 200):
        txt = call("efetch.fcgi", {"db": "pubmed", "id": ",".join(todo[s:s + 200]), "rettype": "medline", "retmode": "text"})
        if not txt:
            continue
        for rec in txt.split("\n\nPMID- ")[0:]:
            rec = rec if rec.startswith("PMID- ") else "PMID- " + rec
            pm = re.search(r"^PMID- (\d+)", rec, re.M)
            if not pm:
                continue
            mesh = [x.strip() for x in re.findall(r"^MH  - (.+)$", rec, re.M)]
            pts = [x.strip() for x in re.findall(r"^PT  - (.+)$", rec, re.M)]
            dp = re.search(r"^DP  - (\d{4})", rec, re.M)
            f.write(json.dumps({"pmid": pm.group(1), "mesh": mesh, "pt": pts,
                                "year": int(dp.group(1)) if dp else None}) + "\n")
        if (s // 200) % 50 == 0:
            print("efetch", s, flush=True)
            f.flush()
print("DONE")
