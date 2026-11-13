"""Find labeled tickets similar to a new ticket with BM25 (pure Python, no extra packages).

BM25 scores a document higher when it shares the query's words, more so for rare words
(IDF), with diminishing returns for repeats (k1) and a penalty for long documents (b).
Used for dynamic few-shot: the most similar labeled tickets become the prompt's examples.
"""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass

from capstone_project.records import Ticket
from capstone_project.text import clean_text, ticket_text, tokenize

STOPWORDS = frozenset(
    "a an and are as at be been but by can could dear do for from had has have hello hi "
    "hope i in into is it its me message my of on or our please reaching regarding support "
    "team thank thanks that the their this to us we well with would you your customer".split()
)


def _terms(ticket: Ticket) -> list[str]:
    return [w for w in tokenize(ticket_text(ticket)) if w not in STOPWORDS and len(w) > 1]


class BM25Index:
    def __init__(self, tickets: list[Ticket], k1: float = 1.5, b: float = 0.75):
        self.tickets = tickets
        self.k1, self.b = k1, b
        self.docs = [Counter(_terms(t)) for t in tickets]
        self.lengths = [sum(d.values()) for d in self.docs]
        self.avg_len = sum(self.lengths) / max(len(self.lengths), 1)
        df = Counter(term for d in self.docs for term in d)
        n = len(tickets)
        self.idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}

    def scores(self, query: Ticket) -> list[float]:
        q = set(_terms(query))
        out = []
        for doc, length in zip(self.docs, self.lengths):
            norm = self.k1 * (1 - self.b + self.b * length / self.avg_len)
            out.append(sum(
                self.idf[t] * doc[t] * (self.k1 + 1) / (doc[t] + norm)
                for t in q if t in doc
            ))
        return out

    def top_k(self, query: Ticket, k: int) -> list[Ticket]:
        """The k most similar tickets, best first. A pool ticket with the same body is skipped."""
        body = clean_text(query.body)
        ranked = sorted(range(len(self.tickets)), key=self.scores(query).__getitem__, reverse=True)
        hits = []
        for i in ranked:
            if clean_text(self.tickets[i].body) == body:
                continue  # never show the model its own answer
            hits.append(self.tickets[i])
            if len(hits) == k:
                break
        return hits


@dataclass(frozen=True)
class Example:
    """A retrieved ticket with its rank (1 = most similar) and word overlap with the query."""
    ticket: Ticket
    rank: int
    overlap: float  # Jaccard overlap of the two tickets' word sets, 0..1


def word_overlap(a: Ticket, b: Ticket) -> float:
    """Shared distinct words / all distinct words (Jaccard). ~1.0 = near-duplicate wording."""
    x, y = set(_terms(a)), set(_terms(b))
    return len(x & y) / len(x | y) if x | y else 0.0


def ranked_examples(index: "BM25Index", query: Ticket, k: int) -> list[Example]:
    """BM25 top-k, annotated with rank and word overlap so the model can tell how close each is."""
    return [Example(t, rank, round(word_overlap(query, t), 2))
            for rank, t in enumerate(index.top_k(query, k), start=1)]


def near_duplicate_queue(examples: list[Example], threshold: float) -> str | None:
    """The rank-1 example's queue if its word overlap with the ticket is >= threshold, else None.

    A rank-1 example with very high overlap is almost certainly another version of the same
    ticket, so its label is more reliable than any judgment call (code decides, not the LLM).
    """
    if examples and examples[0].rank == 1 and examples[0].overlap >= threshold:
        return examples[0].ticket.queue
    return None


def remove_overlap(pool: list[Ticket], held_out: list[Ticket]) -> tuple[list[Ticket], int]:
    """Drop pool tickets whose body also appears in the held-out tickets (eval set, static examples)."""
    banned = {clean_text(t.body) for t in held_out}
    kept = [t for t in pool if clean_text(t.body) not in banned]
    return kept, len(pool) - len(kept)
