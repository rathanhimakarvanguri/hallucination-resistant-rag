"""Run the baseline systems on every question in data/questions.csv.

    python run_baseline.py                  # B0 and B1
    python run_baseline.py --systems B0 B1 S1

Writes results/run_<timestamp>.csv with one row per (question, system) and empty
columns for manual labelling:
    label_correct      yes / partly / no
    label_hallucinated yes / no   (any statement not supported by the regulation)
    label_type         invented_source / wrong_detail / outdated_rule / other
    notes
"""
import argparse
import csv
import time
from datetime import datetime

from src import config
from src.pipelines import SYSTEMS
from src.text import cite


def load_questions():
    with open(config.QUESTIONS_FILE, encoding="utf-8") as f:
        return [r for r in csv.DictReader(f) if r.get("question", "").strip()]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--systems", nargs="+", default=["B0", "B1"], choices=SYSTEMS)
    p.add_argument("--limit", type=int, default=None, help="only the first N questions")
    args = p.parse_args()

    questions = load_questions()[: args.limit]
    retriever = None
    if any(s != "B0" for s in args.systems):
        from src.retrieve import Retriever
        retriever = Retriever()

    config.RESULTS_DIR.mkdir(exist_ok=True)
    out_path = config.RESULTS_DIR / f"run_{datetime.now():%Y%m%d_%H%M}.csv"
    fields = ["qid", "type", "question", "gold_answer", "gold_doc", "gold_clause", "system", "answer",
              "retrieved_clauses", "gold_clause_retrieved", "seconds",
              "label_correct", "label_hallucinated", "label_type", "notes"]

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for n, q in enumerate(questions, 1):
            for s in args.systems:
                t0 = time.time()
                out = SYSTEMS[s](q["question"], retriever)
                got = [(c["doc_id"], c["clause"]) for c in out["retrieved"]]
                gold_hit = ""
                if out["retrieved"] and q.get("gold_clause"):
                    gold_hit = "yes" if any(d == q.get("gold_doc") and cl == q["gold_clause"] for d, cl in got) else "no"
                w.writerow({
                    "qid": q.get("qid", n), "type": q.get("type", ""), "question": q["question"],
                    "gold_answer": q.get("gold_answer", ""), "gold_doc": q.get("gold_doc", ""),
                    "gold_clause": q.get("gold_clause", ""), "system": s, "answer": out["answer"],
                    "retrieved_clauses": " ; ".join(cite(c) for c in out["retrieved"]),
                    "gold_clause_retrieved": gold_hit, "seconds": round(time.time() - t0, 1),
                })
                f.flush()
                print(f"[{n}/{len(questions)}] {s} done ({time.time() - t0:.0f}s)")
    print(f"\nSaved -> {out_path}\nOpen it in Excel and fill the label_* columns.")


if __name__ == "__main__":
    main()
