"""Completion-aware numeric scoring mechanics, not model benchmark scores."""

import pytest

from equylaptatemps.completion import complete_answer, final_answer_stop, summarize_completed_math


@pytest.mark.parametrize(
    "text,reason,expected",
    [
        ("First, each child requires 5", "max_new_tokens", None),
        ("#### 12", "max_new_tokens", None),
        ("#### 12", "eos", "12"),
        ("#### 12\n", "completed_answer", "12"),
        (r"Thus \boxed{1/2}", "max_new_tokens", "0.5"),
        ("#### $1,250.0\n", "completed_answer", "1.25E+3"),
        (r"\boxed{2} followed by #### 3" + "\n", "completed_answer", "3"),
        ("#### 2+3\n", "completed_answer", None),
        (r"\boxed{1/0}", "max_new_tokens", None),
    ],
)
def test_complete_answer_requires_final_marker_and_closed_termination(text, reason, expected):
    assert complete_answer(text, reason) == expected


def test_do_not_stop_mid_numeric_generation():
    assert not final_answer_stop("#### 1")
    assert final_answer_stop("#### 12\n")
    assert final_answer_stop(r"\boxed{12}")


def test_all_completed_or_truncated_cases_stay_in_denominator():
    rows = [
        {
            "reference": "#### 2",
            "response": "#### 2\n",
            "stop_reason": "completed_answer",
            "prediction": "2",
            "correct": True,
        },
        {
            "reference": "#### 5",
            "response": "Each child requires 5",
            "stop_reason": "max_new_tokens",
            "prediction": None,
            "correct": False,
        },
    ]
    result = summarize_completed_math(rows)
    assert result["accuracy"] == 0.5 and result["completed_fraction"] == 0.5
    rows[1]["correct"] = True
    with pytest.raises(ValueError, match="Substituted"):
        summarize_completed_math(rows)
    with pytest.raises(ValueError):
        summarize_completed_math([])
