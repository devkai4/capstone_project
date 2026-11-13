"""Week 3: how predictable are the labels from the ticket text at all? (no LLM, no cost)

    uv run capstone-ingest -n 2100 --out data/pool_2100.json      # once
    uv run python scripts/week3_baseline.py --pool data/pool_2100.json

Three non-LLM classifiers are trained on the pool (minus every ticket in tickets_100.json)
and scored on the same 50 eval tickets as the LLM, plus 5-fold cross-validation on the
pool itself (more tickets -> a steadier number than 50):

    majority     always answer the most common label (the floor)
    bm25-knn     vote of the k most similar pool tickets (what dynamic few-shot can copy)
    tfidf-logreg TF-IDF word features + logistic regression (a classic text classifier)

If even a trained classifier stays far below 70% on queue, the gap is mostly label noise.
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline

from capstone_project.evaluation import LABELS, accuracy
from capstone_project.records import Ticket, load_json
from capstone_project.retrieval import BM25Index, remove_overlap
from capstone_project.text import ticket_text


@dataclass(frozen=True)
class Pred:
    queue: str
    priority: str
    type: str


def majority(pool: list[Ticket], n: int) -> list[Pred]:
    top = {label: Counter(getattr(t, label) for t in pool).most_common(1)[0][0] for label in LABELS}
    return [Pred(**top)] * n


def bm25_knn(pool: list[Ticket], tickets: list[Ticket], k: int) -> list[Pred]:
    index = BM25Index(pool)
    preds = []
    for t in tickets:
        hits = index.top_k(t, k)
        preds.append(Pred(**{
            label: Counter(getattr(h, label) for h in hits).most_common(1)[0][0] for label in LABELS
        }))
    return preds


def tfidf_logreg():
    return make_pipeline(
        TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True, stop_words="english"),
        LogisticRegression(max_iter=2000, C=4.0),
    )


def logreg_preds(pool: list[Ticket], tickets: list[Ticket]) -> list[Pred]:
    x_pool = [ticket_text(t) for t in pool]
    x_eval = [ticket_text(t) for t in tickets]
    columns = {}
    for label in LABELS:
        model = tfidf_logreg().fit(x_pool, [getattr(t, label) for t in pool])
        columns[label] = model.predict(x_eval)
    return [Pred(*row) for row in zip(*(columns[label] for label in LABELS))]


def logreg_cv(pool: list[Ticket], folds: int = 5) -> dict[str, float]:
    x = [ticket_text(t) for t in pool]
    out = {}
    for label in LABELS:
        y = [getattr(t, label) for t in pool]
        # a class needs >= `folds` members for stratified splits; rare classes are merged
        counts = Counter(y)
        y = [v if counts[v] >= folds else "(rare)" for v in y]
        cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=0)
        out[label] = cross_val_score(tfidf_logreg(), x, y, cv=cv).mean()
    out["mean"] = sum(out[label] for label in LABELS) / len(LABELS)
    return out


def row(name: str, acc: dict[str, float]) -> str:
    return (f"| {name:<22} | {acc['queue']:>5.0%} | {acc['priority']:>5.0%} "
            f"| {acc['type']:>5.0%} | {acc['mean']:>5.0%} |")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data", type=Path, default=Path("data/tickets_100.json"))
    parser.add_argument("--pool", type=Path, required=True)
    parser.add_argument("-n", type=int, default=50)
    parser.add_argument("--start", type=int, default=0, help="first eval ticket (50 = held-out set)")
    args = parser.parse_args()

    all_100 = load_json(args.data)
    tickets = all_100[args.start: args.start + args.n]
    pool, dropped = remove_overlap(load_json(args.pool), all_100)
    print(f"pool: {len(pool)} tickets ({dropped} removed because they are in {args.data.name}); "
          f"eval: tickets {args.start}-{args.start + len(tickets) - 1}")
    print("pool queue distribution:", dict(Counter(t.queue for t in pool).most_common()))

    header = "| method                 | queue | prio  | type  | mean  |\n|---|---|---|---|---|"
    print(f"\non the {args.n} eval tickets (same set as the LLM):")
    print(header)
    print(row("majority", accuracy(tickets, majority(pool, len(tickets)))))
    for k in (1, 5, 10):
        print(row(f"bm25-knn k={k}", accuracy(tickets, bm25_knn(pool, tickets, k))))
    print(row("tfidf-logreg", accuracy(tickets, logreg_preds(pool, tickets))))

    print(f"\n5-fold cross-validation on the pool ({len(pool)} tickets, steadier estimate):")
    print(header)
    print(row("tfidf-logreg (cv)", logreg_cv(pool)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
