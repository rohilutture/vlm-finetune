import pytest

from vlm_lab.stats import paired_bootstrap_diff, wilson
from vlm_lab.verify import (accuracy_reward, extract_answer, format_reward, has_single_answer_tag, normalize,
                            total_reward)


@pytest.mark.parametrize("text,want", [
    ("<answer>3</answer>", "3"),
    ("I count them. <answer> 03 </answer>", "3"),
    ("<answer>Yes</answer>", "yes"),
    ("<answer>b</answer>", "B"),
    ("<answer>4.</answer>", "4"),
    ("first <answer>1</answer> wait, actually <answer>2</answer>", "2"),   # the LAST tag counts
    ("The answer is 3", None),
    ("<answer>3", None),
    ("", None),
    ("<ANSWER>7</ANSWER>", "7"),
])
def test_extract_and_normalize(text, want):
    assert extract_answer(text) == want


def test_accuracy_is_exact_after_normalization():
    assert accuracy_reward("x <answer>3</answer>", "3") == 1.0
    assert accuracy_reward("x <answer>03</answer>", "3") == 1.0
    assert accuracy_reward("x <answer>4</answer>", "3") == 0.0
    assert accuracy_reward("answer is 3", "3") == 0.0              # no tag: no credit for a right guess
    assert accuracy_reward("<answer>3 apples</answer>", "3") == 0.0  # not a bare answer


def test_format_reward_requires_exactly_one_closing_tag_at_the_end():
    assert format_reward("reason <answer>1</answer>") == 1.0
    assert format_reward("reason <answer>1</answer> trailing text") == 0.0
    assert format_reward("<answer>1</answer> <answer>2</answer>") == 0.0
    assert has_single_answer_tag("<answer>x</answer>")


def test_total_reward_never_lets_format_outrank_correctness():
    right = total_reward("ok <answer>3</answer>", "3")
    wrong_formatted = total_reward("ok <answer>4</answer>", "3")
    wrong_unformatted = total_reward("4", "3")
    assert right == 1.0 and 0 < wrong_formatted < right and wrong_unformatted == 0.0
    # the reward-hacking guard from the GRPO text-to-SQL project: a well-formed empty answer must not beat correct
    assert total_reward("<answer></answer>", "3") <= 0.1


def test_normalize_leaves_unknown_text_lowercased():
    assert normalize("Maybe") == "maybe"


def test_wilson_interval_properties():
    lo, hi = wilson(50, 100)
    assert lo < 0.5 < hi and hi - lo < 0.21
    assert wilson(0, 10)[0] == 0.0 and wilson(10, 10)[1] == 1.0
    wide, narrow = wilson(5, 10), wilson(500, 1000)
    assert (wide[1] - wide[0]) > (narrow[1] - narrow[0])
    assert wilson(0, 0) == (0.0, 1.0)


def test_paired_bootstrap_detects_real_and_null_differences():
    a = [1.0] * 70 + [0.0] * 30     # 70 percent
    b = [1.0] * 50 + [0.0] * 50     # 50 percent, paired by position (a superset on the first 50)
    mean, lo, hi = paired_bootstrap_diff(a, b, iters=2000)
    assert abs(mean - 0.2) < 1e-9 and lo > 0.1 and hi < 0.3
    same_mean, lo, hi = paired_bootstrap_diff(a, a, iters=500)
    assert same_mean == 0 and lo == 0 == hi
    noisy_a = [1.0, 0.0] * 20
    noisy_b = [0.0, 1.0] * 20          # identical accuracy, different examples: the interval must straddle 0
    m, lo, hi = paired_bootstrap_diff(noisy_a, noisy_b, iters=2000)
    assert lo < 0 < hi
