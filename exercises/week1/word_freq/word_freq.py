"""Word-frequency counter for ticket text (Think Python ch. 9-11: lists, dicts, tuples)."""

import string
import sys

# Common words that carry no meaning for ticket triage
STOP_WORDS = {
    "the", "a", "an", "to", "of", "in", "on", "for", "with", "is", "are",
    "was", "and", "or", "but", "not", "that", "this", "it", "at", "how",
    "why", "does", "did", "even", "though", "after", "before", "because",
    "when", "now", "more", "than", "says", "asks", "wants", "know", "needs",
}


def read_words(filename):
    """Return a list of cleaned, lowercase words from the file (ch. 9: lists)."""
    words = []
    with open(filename, encoding="utf-8") as f:
        for line in f:
            for token in line.split():
                word = token.strip(string.punctuation).lower()
                if word:
                    words.append(word)
    return words


def count_words(words, skip_stop_words=True):
    """Return a dict mapping word -> count (ch. 10: dicts)."""
    counts = {}
    for word in words:
        if skip_stop_words and word in STOP_WORDS:
            continue
        counts[word] = counts.get(word, 0) + 1
    return counts


def sort_key(item):
    """Sort by count descending, then word alphabetically (ch. 11: tuples)."""
    word, count = item
    return (-count, word)


def top_words(counts, n=10):
    """Return the n most frequent (word, count) tuples."""
    items = list(counts.items())
    items.sort(key=sort_key)
    return items[:n]


def main():
    filename = sys.argv[1] if len(sys.argv) > 1 else "tickets.txt"
    words = read_words(filename)
    counts = count_words(words)

    print(f"Total words: {len(words)}, unique (excluding stop words): {len(counts)}")
    print()
    print(f"{'Rank':<5}{'Word':<15}{'Count':>5}")
    for rank, (word, count) in enumerate(top_words(counts), start=1):
        print(f"{rank:<5}{word:<15}{count:>5}")


if __name__ == "__main__":
    main()