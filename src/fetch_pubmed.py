"""P04 S6 / S8: PubMed records from NCBI.

python fetch_pubmed.py indexing   For every PMID in data/mesh.jsonl: MEDLINE indexing method (IndexingMethod
                                  attribute of MedlineCitation: absent = manual, "Curated" = automated then
                                  reviewed, "Automated"), publication year, MeSH headings. E-utilities efetch,
                                  200 PMIDs per request, <= 3 requests/s (no API key). -> data/pubmed_indexing.jsonl
python fetch_pubmed.py baseline N PubMed annual baseline files (https://ftp.ncbi.nlm.nih.gov/pubmed/baseline/), N files
                                  drawn at random (seed 0) across the PMID range; records with an abstract and MeSH
                                  -> data/pubmed_sample.jsonl.gz {pmid, title, abstract, mesh, pt, year}
"""

from pathlib import Path

WORKSPACE_ROOT = Path(__file__).resolve().parents[1].as_posix()
import gzip
import io
import json
import os
import random
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import requests
from lxml import etree

ROOT = WORKSPACE_ROOT
S = requests.Session()
S.headers["User-Agent"] = "species-scope-retrieval-study/0.1 (academic research; python-requests)"


def parse_article(el):
    mc = el.find("MedlineCitation")
    if mc is None:
        return None
    pmid = mc.findtext("PMID")
    art = mc.find("Article")
    title = "".join(art.find("ArticleTitle").itertext()) if art is not None and art.find("ArticleTitle") is not None else ""
    abst = " ".join("".join(a.itertext()) for a in art.findall("Abstract/AbstractText")) if art is not None else ""
    mesh = [mh.findtext("DescriptorName") for mh in mc.findall("MeshHeadingList/MeshHeading")]
    pt = [p.text for p in art.findall("PublicationTypeList/PublicationType")] if art is not None else []
    y = mc.findtext("DateCompleted/Year") or (art.findtext("Journal/JournalIssue/PubDate/Year") if art is not None else None)
    return {"pmid": pmid, "indexing": mc.get("IndexingMethod") or "Manual", "status": mc.get("Status"), "year": int(y) if y and y.isdigit() else None,
            "title": title, "abstract": abst, "mesh": [m for m in mesh if m], "pt": [p for p in pt if p]}


def efetch(ids):
    for attempt in range(8):
        try:
            r = S.post("https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi",
                       data={"db": "pubmed", "id": ",".join(ids), "retmode": "xml"}, timeout=120)
            if r.status_code == 200 and r.content.strip().startswith(b"<?xml"):
                root = etree.fromstring(r.content)
                return [x for x in (parse_article(a) for a in root.findall("PubmedArticle")) if x]
            time.sleep(3 * (attempt + 1))
        except Exception as e:  # noqa: BLE001
            print("retry", attempt, str(e)[:100], flush=True); time.sleep(3 * (attempt + 1))
    return []


def indexing():
    out = f"{ROOT}/data/pubmed_indexing.jsonl"
    pm = [json.loads(l)["pmid"] for l in open(f"{ROOT}/data/mesh.jsonl")]
    done = {json.loads(l)["pmid"] for l in open(out)} if os.path.exists(out) else set()
    todo = [p for p in dict.fromkeys(pm) if p and p not in done]
    batches = [todo[i:i + 200] for i in range(0, len(todo), 200)]
    print("pmids", len(todo), "batches", len(batches), flush=True)
    t0 = time.time()
    with open(out, "a") as f, ThreadPoolExecutor(3) as ex:
        for k, recs in enumerate(ex.map(lambda b: (time.sleep(0.35), efetch(b))[1], batches)):
            for r in recs:
                f.write(json.dumps({x: r[x] for x in ("pmid", "indexing", "status", "year", "mesh")}) + "\n")
            if k % 50 == 0:
                print(k, len(batches), f"{time.time() - t0:.0f}s", flush=True)
    print("INDEXING_DONE", flush=True)


def baseline(n):
    idx = S.get("https://ftp.ncbi.nlm.nih.gov/pubmed/baseline/", timeout=60).text
    files = sorted(set(re.findall(r"(pubmed\d+n\d+\.xml\.gz)\"", idx)))
    pick = sorted(random.Random(0).sample(files, n))
    print("baseline files", len(files), "picked", n, flush=True)
    out = f"{ROOT}/data/pubmed_sample.jsonl.gz"
    done = set()
    if os.path.exists(f"{out}.files"):
        done = set(open(f"{out}.files").read().split())
    fo = gzip.open(out, "at")
    for i, fn in enumerate(pick):
        if fn in done:
            continue
        for attempt in range(5):
            try:
                r = S.get("https://ftp.ncbi.nlm.nih.gov/pubmed/baseline/" + fn, timeout=600)
                r.raise_for_status()
                break
            except Exception as e:  # noqa: BLE001
                print("retry", fn, str(e)[:80], flush=True); time.sleep(10)
        k = 0
        for _, el in etree.iterparse(io.BytesIO(gzip.decompress(r.content)), tag="PubmedArticle"):
            a = parse_article(el)
            el.clear()
            if a and a["abstract"] and a["mesh"]:
                fo.write(json.dumps({x: a[x] for x in ("pmid", "title", "abstract", "mesh", "pt", "year", "indexing")}) + "\n"); k += 1
        fo.flush()
        with open(f"{out}.files", "a") as g:
            g.write(fn + "\n")
        print(i, fn, k, flush=True)
    print("BASELINE_DONE", flush=True)


if __name__ == "__main__":
    if sys.argv[1] == "indexing":
        indexing()
    else:
        baseline(int(sys.argv[2]))
