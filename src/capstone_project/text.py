"""Clean ticket text before it goes to counting, retrieval or a model."""

from __future__ import annotations

import re
import unicodedata

from capstone_project.records import Ticket

_WHITESPACE = re.compile(r"\s+")
_WORD = re.compile(r"[a-z0-9]+(?:['-][a-z0-9]+)*")


def clean_text(text: str | None, *, lower: bool = False) -> str:
    """Normalize one piece of text.

    1. None / "" -> ""
    2. Unicode NFKC (full-width "ＡＢＣ１" -> "ABC1", non-breaking space -> space)
    3. Drop invisible format chars (zero-width space etc.), turn control chars into spaces
    4. Collapse every run of whitespace (incl. newlines) into one space, strip the ends
    5. Optionally lowercase
    """
    if not text:
        return ""
    text = unicodedata.normalize("NFKC", text)
    chars = []
    for ch in text:
        category = unicodedata.category(ch)
        if category == "Cf":  # format chars, e.g. U+200B zero-width space
            continue
        chars.append(" " if category.startswith("C") else ch)
    text = _WHITESPACE.sub(" ", "".join(chars)).strip()
    return text.lower() if lower else text


def ticket_text(ticket: Ticket) -> str:
    """The text the triage agent reads: cleaned subject + cleaned body."""
    parts = [clean_text(ticket.subject), clean_text(ticket.body)]
    return "\n\n".join(part for part in parts if part)


def tokenize(text: str | None) -> list[str]:
    """Lowercase words (letters/digits, keeping inner ' and -), e.g. "Can't log-in!" -> ["can't", "log-in"]."""
    return _WORD.findall(clean_text(text, lower=True))
