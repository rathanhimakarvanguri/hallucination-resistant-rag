"""The systems compared in the dissertation.

B0  LLM only, no retrieval
B1  Basic RAG: dense retrieval, top-k clauses, plain prompt        (main baseline)
S1  B1 + hybrid retrieval (BM25 + dense, RRF)                      (first improvement)
S2-S4 (citations, claim verification, abstention) come after the viva.
"""
from src import config, llm
from src.text import cite

B0_SYSTEM = ("You are an expert on Indian financial regulation (RBI and SEBI). "
             "Answer the user's question accurately and concisely. "
             "Mention the relevant circular or Master Direction and paragraph if you know it.")

RAG_SYSTEM = ("You are an assistant that answers questions about Indian financial regulation (RBI and SEBI) "
              "using the context provided. Answer concisely.")


def format_context(chunks):
    return "\n\n".join(f"{cite(c)}\n{c['text']}" for c in chunks)


def run_b0(question, retriever=None):
    answer = llm.chat(B0_SYSTEM, question)
    return {"system": "B0", "answer": answer, "retrieved": []}


def _rag(question, retriever, mode, name):
    chunks = retriever.search(question, k=config.TOP_K, mode=mode)
    user = f"Context:\n{format_context(chunks)}\n\nQuestion: {question}"
    answer = llm.chat(RAG_SYSTEM, user)
    return {"system": name, "answer": answer, "retrieved": chunks}


def run_b1(question, retriever):
    return _rag(question, retriever, "dense", "B1")


def run_s1(question, retriever):
    return _rag(question, retriever, "hybrid", "S1")


SYSTEMS = {"B0": run_b0, "B1": run_b1, "S1": run_s1}
