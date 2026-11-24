"""Central settings for the RBI/SEBI hallucination-resistant RAG project."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw"            # put PDFs in data/raw/rbi and data/raw/sebi
CHUNKS_FILE = ROOT / "data" / "processed" / "chunks.jsonl"
INDEX_DIR = ROOT / "data" / "index"
QUESTIONS_FILE = ROOT / "data" / "questions.csv"
RESULTS_DIR = ROOT / "results"

# Ollama (runs locally on the Mac)
OLLAMA_URL = "http://localhost:11434"
CHAT_MODEL = "llama3.1:8b"          # alternative: "qwen2.5:7b"
EMBED_MODEL = "nomic-embed-text"

# Chunking
MAX_CHUNK_CHARS = 1500     # a clause longer than this is split
MIN_CHUNK_CHARS = 60       # shorter pieces: headings join the next clause, fragments the previous

# Retrieval
TOP_K = 5                  # clauses passed to the LLM
CANDIDATES = 20            # candidates taken from each retriever before fusion
RRF_K = 60                 # reciprocal-rank-fusion constant

TEMPERATURE = 0.0
