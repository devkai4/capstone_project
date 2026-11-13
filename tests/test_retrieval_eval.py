import pytest

from capstone_project.classifier import user_message
from capstone_project.evaluation import accuracy, percentile
from capstone_project.records import Ticket
from capstone_project.retrieval import BM25Index, remove_overlap


def t(subject, body, queue="Technical Support", priority="high", type_="Incident"):
    return Ticket(subject=subject, body=body, queue=queue, priority=priority, type=type_,
                  language="en")


POOL = [
    t("VPN keeps dropping", "Our VPN connection drops every hour on the router.", "IT Support"),
    t("Invoice wrong", "The March invoice charges us twice for the same licence.",
      "Billing and Payments", "low", "Request"),
    t("Printer offline", "The office printer is offline after the firmware update."),
]


# ---------- BM25 retrieval ----------

def test_top_k_ranks_the_most_similar_ticket_first():
    query = t("VPN problem", "The VPN on our router disconnects again and again.")
    assert BM25Index(POOL).top_k(query, 1)[0].subject == "VPN keeps dropping"


def test_top_k_never_returns_the_query_itself():
    hits = BM25Index(POOL).top_k(POOL[1], 3)
    assert POOL[1] not in hits and len(hits) == 2


def test_remove_overlap_drops_held_out_bodies():
    held_out = [t("same body, other subject", POOL[0].body)]
    kept, dropped = remove_overlap(POOL, held_out)
    assert dropped == 1 and POOL[0] not in kept


# ---------- dynamic few-shot message ----------

def test_user_message_with_examples_puts_labels_before_the_ticket():
    msg = user_message(POOL[2], examples=[POOL[1]])
    assert msg.index("<examples>") < msg.index("Classify this ticket:") < msg.rindex("<ticket>")
    assert '<answer>{"queue": "Billing and Payments", "priority": "low", "type": "Request"}</answer>' in msg


def test_user_message_without_examples_is_just_the_ticket():
    assert user_message(POOL[2]).startswith("<ticket>")


# ---------- metrics ----------

def test_accuracy_per_label_and_mean():
    truth = POOL
    preds = [POOL[0], POOL[0], None]  # 1st: all labels right, 2nd: all wrong, 3rd: invalid output
    acc = accuracy(truth, preds)
    assert acc == pytest.approx({"queue": 1 / 3, "priority": 1 / 3, "type": 1 / 3, "mean": 1 / 3})


@pytest.mark.parametrize(("q", "expected"), [(50, 3.0), (95, 5.0), (100, 5.0)])
def test_percentile_nearest_rank(q, expected):
    assert percentile([5.0, 1.0, 3.0, 2.0, 4.0], q) == expected


# ---------- v3: ranked examples with word overlap ----------

def test_word_overlap_is_jaccard_of_word_sets():
    from capstone_project.retrieval import word_overlap

    a = t("x", "vpn router drops")
    assert word_overlap(a, a) == 1.0
    # 1-letter words are ignored: {vpn, router} shared / {vpn, router, drops, works}
    assert word_overlap(a, t("x", "vpn router works")) == pytest.approx(2 / 4)


def test_ranked_examples_keep_bm25_order_and_number_from_one():
    from capstone_project.retrieval import ranked_examples

    query = t("VPN problem", "The VPN on our router disconnects again and again.")
    examples = ranked_examples(BM25Index(POOL), query, 2)
    assert [e.rank for e in examples] == [1, 2]
    assert examples[0].ticket.subject == "VPN keeps dropping"
    assert examples[0].overlap > examples[1].overlap


def test_user_message_tags_ranked_examples():
    from capstone_project.retrieval import Example

    msg = user_message(POOL[2], examples=[Example(POOL[0], 1, 0.83)])
    assert '<example rank="1" overlap="0.83">' in msg


# ---------- near-duplicate rule ----------

@pytest.mark.parametrize(("overlap", "expected"), [(0.65, "IT Support"), (0.40, None)])
def test_near_duplicate_queue(overlap, expected):
    from capstone_project.retrieval import Example, near_duplicate_queue

    assert near_duplicate_queue([Example(POOL[0], 1, overlap)], threshold=0.5) == expected


def test_near_duplicate_queue_needs_examples():
    from capstone_project.retrieval import near_duplicate_queue

    assert near_duplicate_queue([], threshold=0.5) is None
