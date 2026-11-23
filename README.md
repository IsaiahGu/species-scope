# You Can Ask for Mice but Not for Humans
**The Unmarked Study Population in Biomedical Retrieval**

The audit measures how population phrases change the share of human-only and animal-only articles among the top ten retrieved abstracts. It then tests whether answers disclose the population behind a finding.

## Inspect the paper results

[Figure 2 source table](reference/fig2_heatmap.csv) gives each phrase, the neutral-adjusted shifts, and counts of significant changes across 33 retriever–collection pairs. The other CSVs in [reference](reference/) report collection sizes, term prevalence, exposure and answer-level remedies. In the heatmap table, positive shifts indicate an increase in the human-only share, including for animal-pointing phrases.

## Run the retrieval audit

Use Linux and Python 3.10 or newer. From the repository root, install a CUDA-compatible PyTorch build for neural retrieval, then run:

```bash
python -m pip install -r requirements.txt
mkdir -p data/beir models results logs
hf download --repo-type dataset rag-datasets/rag-mini-bioasq --local-dir data/rag-mini-bioasq
for ds in nfcorpus trec-covid; do
  hf download --repo-type dataset BeIR/$ds --local-dir data/beir/$ds
  hf download --repo-type dataset BeIR/$ds-qrels --local-dir data/beir/$ds/qrels
done
MODELS=bm25 python src/sweep.py bioasq
```

This first run writes BioASQ rankings to `results/runs_sweep_bioasq.parquet`. The full sweep writes to the same filename. To generate the scope labels, place the CORD-19 metadata snapshot dated 2020-07-16 at `data/cord19_metadata_2020-07-16.csv`, then run `python src/fetch_mesh.py`. It maps document IDs to PMIDs and retrieves MeSH tags through NCBI E-utilities.

For the neural runs, download the checkpoints named by [the encoder loader](shared/embed.py) under `models/`. The primary directories are `e5-base-v2`, `qwen3-emb-0.6b`, `medcpt-query`, `medcpt-article`, `medcpt-cross`, `harrier-oss-v1-0.6b` and `embeddinggemma-2`. The MedCPT directories hold the NCBI query encoder, article encoder and cross-encoder respectively. Use the same three dataset arguments in each ranking stage.

```bash
python src/run_retrieval.py bioasq nfcorpus trec-covid
python src/sweep.py bioasq nfcorpus trec-covid
for family in splade colbert qwen4b fusion; do
  python src/families.py "$family" bioasq nfcorpus trec-covid
done
python src/analyze.py
python src/sweep_analyze.py
```

The additional families use prithivida/Splade_PP_en_v1, lightonai/GTE-ModernColBERT-v1 and Qwen/Qwen3-Embedding-4B in `models/splade-pp`, `models/gte-moderncolbert` and `models/qwen3-emb-4b`. Install PyLate for the ColBERT stage. Fusion uses the saved rankings.

## Answer and robustness experiments

`analyze.py` uses the local classifier and the two API judges. Configure an OpenAI-compatible endpoint using `LOCAL_LLM_URL` (default localhost port 8000, model name qwen3.5-9b), plus `OPENAI_BASE_URL` and `OPENAI_API_KEY` for remote models. Supply the endpoints used by your experiment.

`instruct.py` runs instruction variants and `scale.py` evaluates a larger PubMed corpus prepared by `fetch_pubmed.py`. After building the required retrieval runs, `answers.py contexts` prepares the evidence sets, `answers.py answer <reader>` generates cited answers and `answers.py judge <judge> answers` judges them. Run these scripts with Python from `src/` or prefix their filenames with `src/`. See `s5_analysis.py` for paired answer-level effects.


