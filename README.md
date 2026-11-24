# Hallucination-Resistant RAG for RBI and SEBI Regulatory QA

M.Tech AIML dissertation (BITS WILP, AIMLC ZG628T) by Rathan Himakar Vanguri.

The goal is to measure and reduce hallucination when LLMs answer questions on RBI Master Directions and SEBI Master Circulars.

## Systems

| ID | System | Status |
|----|--------|--------|
| B0 | LLM only, no retrieval | done |
| B1 | Basic RAG (dense retrieval, top-5 clauses) | done |
| S1 | B1 + hybrid retrieval (BM25 + dense, reciprocal rank fusion) | done |
| S2 | S1 + clause-level citations | planned |
| S3 | S2 + claim verification | planned |
| S4 | S3 + abstention | planned |

## Setup (macOS, Apple Silicon)

```bash
cd ~/Documents/dissertation-rag
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Ollama app must be running
ollama pull llama3.1:8b          # ~4.9 GB
ollama pull nomic-embed-text     # ~270 MB
```

## Run

```bash
source .venv/bin/activate

# 1. Put PDFs in data/raw/rbi/ and data/raw/sebi/, then split them into clauses
python -m src.ingest

# 2. Build the search indexes (embeddings run through Ollama)
python -m src.index

# 3. Ask a question
python ask.py "How often must KYC be updated for high-risk customers?"
python ask.py "..." --system B0      # LLM only
python ask.py "..." --system S1      # hybrid retrieval

# 4. Run all questions in data/questions.csv through B0 and B1
python run_baseline.py
python run_baseline.py --systems B0 B1 S1 --limit 5
```

Results go to `results/run_<date>.csv`. Open the file in Excel and fill in the `label_*` columns:

- `label_correct`: yes / partly / no
- `label_hallucinated`: yes / no. Yes means any statement that is not supported by the regulation.
- `label_type`: invented_source / wrong_detail / outdated_rule / other

## Project layout

```
src/config.py     settings (models, chunk sizes, top-k)
src/ingest.py     PDF -> clause-level chunks (keeps clause numbers and pages)
src/index.py      embeddings index
src/retrieve.py   BM25, dense and hybrid (RRF) retrieval
src/pipelines.py  B0, B1, S1
ask.py            ask one question
run_baseline.py   run the benchmark and save a CSV for labelling
data/questions.csv  benchmark questions (qid, type, question, gold_answer, gold_doc, gold_clause)
```

Only public RBI and SEBI documents are used. This is a research prototype and does not give legal or compliance advice.
