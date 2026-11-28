"""The scope lint: the sentence appears once in the README, and a paraphrase elsewhere is flagged."""
import io

import pytest

from passing_check.lint_scope import SCOPE_SENTENCE, _overlap_score, _ascii_safe, check_paraphrase, check_readme


def test_scope_sentence_is_the_ledger_text():
    assert SCOPE_SENTENCE == ("This record reads, on one cache at a time, whether a cache reuse check passes and "
                              "whether the reader's predictions, answers and task outcome under that cache differ "
                              "from a fresh read; it does not propose a check, set an acceptance threshold, or "
                              "measure savings or latency.")


def test_readme_needs_exactly_one():
    assert check_readme("no sentence here") == ["README.md: scope sentence appears 0 times, expected exactly 1"]
    assert check_readme(SCOPE_SENTENCE + "\n\n" + SCOPE_SENTENCE) == \
        ["README.md: scope sentence appears 2 times, expected exactly 1"]
    assert check_readme("x\n" + SCOPE_SENTENCE + "\ny") == []


def test_wrapped_verbatim_sentence_counts_as_one():
    wrapped = ("> This record reads, on one cache at a time, whether a cache reuse check passes and whether\n"
               "> the reader's predictions, answers and task outcome under that cache differ from a fresh\n"
               "> read; it does not propose a check, set an acceptance threshold, or measure savings or latency.")
    assert check_readme(wrapped) == []


PARAPHRASE = ("This record reads, on one cache, whether a reuse check passes and whether the reader's predictions "
              "and answers under that cache differ from a fresh read; savings and latency are not measured.")
NEAR_MISS = ("The first registration names the text and the model before any run starts, and its controls are "
             "scored before any outcome is opened.")


def test_anchor_scores_sit_on_either_side_of_the_threshold():
    # These two anchors hold the margin around the threshold (0.45) on this repo's sentence. If either
    # drifts across, recalibrate the threshold; do not tune the anchors.
    assert _overlap_score(PARAPHRASE) >= 0.60
    assert _overlap_score(NEAR_MISS) <= 0.30


def test_paraphrase_is_flagged():
    result = check_paraphrase(PARAPHRASE, "docs/x.md")
    assert len(result) == 1 and "closely paraphrases the scope sentence" in result[0] and PARAPHRASE in result[0]
    assert check_paraphrase(SCOPE_SENTENCE, "docs/x.md") == []


def test_paraphrase_joined_by_arrow_is_also_flagged():
    text = PARAPHRASE.replace("; savings", " → savings")
    result = check_paraphrase(text, "docs/x.md")
    assert len(result) == 1 and text in result[0]


def test_near_miss_that_shares_keywords_stays_unflagged_on_overlap():
    assert check_paraphrase(NEAR_MISS, "ledger/ledger.md") == []


def test_flagged_nonascii_sentence_is_reported_without_raising():
    text = PARAPHRASE.replace("This record", "This record ρ")
    problems = check_paraphrase(text, "docs/rho.md")
    assert len(problems) == 1 and "ρ" in problems[0]
    stream = io.TextIOWrapper(io.BytesIO(), encoding="cp1252", errors="strict")
    with pytest.raises(UnicodeEncodeError):
        print("SCOPE:", problems[0], file=stream)
    print("SCOPE:", _ascii_safe(problems[0]), file=stream)
    stream.flush()
    buf = stream.buffer.getvalue().decode("cp1252")
    assert "SCOPE:" in buf and "latency" in buf and "\\u03c1" in buf


def test_repo_passes():
    from passing_check.lint_scope import main
    assert main() == 0
