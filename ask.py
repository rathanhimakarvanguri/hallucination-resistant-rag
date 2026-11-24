"""Ask one question from the terminal.

    python ask.py "What is the periodicity of KYC updation for high-risk customers?"
    python ask.py "..." --system B0        # LLM only
    python ask.py "..." --system S1        # hybrid retrieval
"""
import argparse

from src.pipelines import SYSTEMS
from src.text import cite


def main():
    p = argparse.ArgumentParser()
    p.add_argument("question")
    p.add_argument("--system", default="B1", choices=SYSTEMS)
    args = p.parse_args()

    retriever = None
    if args.system != "B0":
        from src.retrieve import Retriever
        retriever = Retriever()
    out = SYSTEMS[args.system](args.question, retriever)

    print(f"\n=== {out['system']} answer ===\n{out['answer']}\n")
    if out["retrieved"]:
        print("=== Retrieved clauses ===")
        for c in out["retrieved"]:
            print(f"{cite(c)}  p.{c['page']}  score={c['score']}\n   {c['text'][:200]}...")


if __name__ == "__main__":
    main()
