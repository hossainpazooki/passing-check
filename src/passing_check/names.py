"""What a continuation names, and how two continuations compare.

A handoff in this corpus asks the receiver for a summary of what the agent did. Two summaries can differ
in wording and still name the same things, so three readings are taken on the decoded text:

- the named set: what the text puts in backticks and that reads as a path or an identifier (NAME);
- overlap: the Jaccard index of two named sets (1.0 when both are empty);
- grounding: the share of a continuation's names that occur verbatim in the receiver prompt it follows.

Only backticked spans count. Prose is never searched: "e.g", "file(s)" or a sentence break written
"tests.Then" are not names, and neither is anything inside a chat-template token.
"""
import re

SPAN = re.compile(r"`([^`\n]{1,200})`")
NAME = re.compile(r"[A-Za-z_][\w./-]*")
WORD = re.compile(r"[a-z0-9_]+")
CONSTANTS = frozenset({"None", "True", "False", "NoneType", "self", "null", "true", "false"})


def named(text: str) -> frozenset[str]:
    """Backticked spans, cut at the first "(" and stripped of a trailing dot, that are one path or identifier,
    are not a single plain lowercase word of fewer than four letters, and are not a language constant."""
    out = set()
    for span in SPAN.findall(text):
        candidate = span.strip().split("(", 1)[0].rstrip(".")
        if (NAME.fullmatch(candidate) and (len(candidate) >= 4 or not candidate.islower())
                and candidate not in CONSTANTS):
            out.add(candidate)
    return frozenset(out)


def distinct_words(text: str) -> int:
    """The number of different lowercase runs of letters, digits and underscores in the text."""
    return len(set(WORD.findall(text.lower())))


def cut_at(ids, stop_ids) -> list[int]:
    """The ids before the first stop id; all of them when none occurs."""
    out = []
    for i in ids:
        if int(i) in stop_ids:
            break
        out.append(int(i))
    return out


def jaccard(a: frozenset, b: frozenset) -> float:
    return 1.0 if not a and not b else len(a & b) / len(a | b)


def grounding(names: frozenset, prompt: str) -> float | None:
    """Share of `names` found verbatim in `prompt`; None when there is no name to ground."""
    if not names:
        return None
    return sum(1 for n in names if n in prompt) / len(names)


def first_difference(a, b) -> int:
    """Index of the first position where two id sequences differ; the shorter length if one is a prefix
    of the other; -1 when they are identical."""
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return i
    return -1 if len(a) == len(b) else min(len(a), len(b))


def agreement(a, b, n: int) -> float:
    """Share of the first n positions where two id sequences hold the same id; a missing position counts
    as a disagreement."""
    return sum(1 for i in range(n) if i < len(a) and i < len(b) and a[i] == b[i]) / n


def common_share(a, b) -> float:
    """Length of the longest common subsequence of two id sequences over the longer length. 1.0 for
    identical sequences; unlike positional agreement, one inserted token does not zero it."""
    if not len(a) or not len(b):
        return 1.0 if len(a) == len(b) else 0.0
    row = [0] * (len(b) + 1)
    for x in a:
        diagonal = 0
        for j, y in enumerate(b, 1):
            diagonal, row[j] = row[j], diagonal + 1 if x == y else max(row[j], row[j - 1])
    return row[-1] / max(len(a), len(b))
