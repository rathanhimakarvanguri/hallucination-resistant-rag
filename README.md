# Hallucination-Resistant RAG for RBI and SEBI Regulatory QA

M.Tech AIML dissertation (BITS WILP, AIMLC ZG628T) by Rathan Himakar Vanguri.

LLM assistants answer questions on RBI and SEBI regulations with unsupported or outdated claims, and basic RAG neither reliably prevents this nor knows when to say "I don't know". This project measures how often that happens and builds a pipeline that answers only from the current rule, cites the clause, and abstains when the evidence is missing.

## Pilot results (30 questions)

Llama 3.1 8B, run locally with Ollama. Every answer was labelled by hand against the gold answer and source clause in [`data/questions.csv`](data/questions.csv).

| Measure | B0: LLM only | B1: basic RAG | S1: hybrid RAG |
|---|---|---|---|
| Answers with a hallucination | 30 / 30 (100%) | 3 / 30 (10%) | 5 / 30 (17%) |
| Fully correct answers | 1 / 30 | 27 / 30 | 25 / 30 |
| Said "I don't know" when it should (unanswerable) | 0 / 4 | 3 / 4 | 2 / 4 |
| Superseded-rule questions answered correctly | 0 / 3 | 1 / 3 | 0 / 3 |
| Gold clause in the top 5 retrieved | – | 20 / 25 | 21 / 25 |

What the pilot shows:

- **Without retrieval the model invents sources.** B0 cited non-existent Master Directions, paragraph numbers and quotes in every answer. For example, it said KYC records go to CKYCR "within thirty days"; the rule says 10.
- **Basic RAG fixes most errors, but not the ones that matter most here.** All of B1's and S1's remaining errors are answers taken from **superseded rules** (for example, saying the fraud directions cover Small Finance Banks, based on the 2024 version), or answers given from outside the documents instead of abstaining.
- **Better retrieval alone does not fix this.** Hybrid retrieval (S1) found the gold clause slightly more often, but old and new versions share most of their wording, so both are retrieved. Superseded documents were in the context for 25 to 26 of 30 questions. This motivates version-aware retrieval, claim verification and abstention (S2 to S4).

The pilot is small; the full benchmark targets about 300 questions.

## Systems

| ID | System | Status |
|----|--------|--------|
| B0 | LLM only, no retrieval | done |
| B1 | Basic RAG (dense retrieval, top-5 clauses) | done |
| S1 | B1 + hybrid retrieval (BM25 + dense, reciprocal rank fusion) | done |
| S2 | S1 + version-aware retrieval and clause-level citations | planned |
| S3 | S2 + claim verification against the retrieved clauses | planned |
| S4 | S3 + abstention when evidence is insufficient | planned |

## Corpus

Ten public documents, listed with their source links and status in [`data/documents.csv`](data/documents.csv): five current RBI/SEBI directions and master circulars, and the five versions they replaced (KYC, digital lending, fraud risk management, stock brokers, LODR). The PDFs are not stored in this repository. Download them from the links and save each one under its `doc_id` as the file name (for example `data/raw/rbi/RBI_KYC_CB_2025.pdf`).

## Setup (macOS, Apple Silicon)

```bash
git clone https://github.com/rathanhimakarvanguri/hallucination-resistant-rag.git
cd hallucination-resistant-rag
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
mkdir -p data/raw/rbi data/raw/sebi    # put the PDFs here

# Ollama app must be running
ollama pull llama3.1:8b          # ~4.9 GB
ollama pull nomic-embed-text     # ~270 MB
```

## Run

```bash
source .venv/bin/activate

# 1. Split the PDFs into clause-level chunks (keeps paragraph numbers and pages)
python -m src.ingest

# 2. Build the search index (embeddings run through Ollama)
python -m src.index

# 3. Ask a question
python ask.py "How often must KYC be updated for high-risk customers?"
python ask.py "..." --system B0      # LLM only
python ask.py "..." --system S1      # hybrid retrieval

# 4. Run all benchmark questions
python run_baseline.py                          # B0 and B1
python run_baseline.py --systems S1 --limit 5
```

Results go to `results/run_<date>.csv`, with empty columns for labelling:

- `label_correct`: yes / partly / no
- `label_hallucinated`: yes / no. Yes means any statement that is not supported by the regulation.
- `label_type`: invented_source / wrong_detail / outdated_rule / other

## Project layout

```
src/config.py       settings (models, chunk sizes, top-k)
src/ingest.py       PDF -> clause-level chunks, with document status from documents.csv
src/index.py        embeddings index
src/retrieve.py     BM25, dense and hybrid (RRF) retrieval
src/pipelines.py    B0, B1, S1
ask.py              ask one question
run_baseline.py     run the benchmark and save a CSV for labelling
data/questions.csv  benchmark questions with gold answers and clauses
data/documents.csv  corpus register: title, date, current/superseded, source link
```

## Tools

Python, PyMuPDF, rank-bm25, nomic-embed-text and Llama 3.1 8B via Ollama, NumPy. Planned: a cross-encoder re-ranker, an NLI model or LLM judge for claim checking, RAGAS and a Streamlit demo.

Only public RBI and SEBI documents are used. This is a research prototype and does not give legal or compliance advice.
