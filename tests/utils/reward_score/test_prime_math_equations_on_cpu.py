# Copyright 2026 Individual Contributor: JianhaoYu-AI
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import pytest

from verl.utils.reward_score import default_compute_score, prime_math


@pytest.mark.parametrize(
    "equation",
    [
        r"x - \sqrt{3}y + 2 = 0",
        r"\dfrac{x^{2}}{18} + \dfrac{y^{2}}{9} = 1",
        "S_{max}=-6",
        r"AC = 8 \text{ meters}",
        r"x_{1} = -1 \text{ and } x_{2} = -2",
        "g(0)=-a",
        "f(-2)=-1",
    ],
)
@pytest.mark.parametrize("box", ["boxed", "fbox"])
def test_identical_boxed_equation_is_accepted(equation: str, box: str) -> None:
    """A boxed gold equation must survive answer-marker extraction."""
    response = f"After working through the problem, the answer is $\\{box}{{{equation}}}$."
    correct, _, extracted = prime_math.compute_score(response, equation)
    assert correct is True
    assert extracted == equation


@pytest.mark.parametrize(
    ("answer", "gold"),
    [
        (r"AC = 8 \text{ feet}", r"AC = 8 \text{ meters}"),
        (r"x_{3} = -1 \text{ and } x_{2} = -2", r"x_{1} = -1 \text{ and } x_{2} = -2"),
        ("0g=-a", "g(0)=-a"),
        ("-2f=-1", "f(-2)=-1"),
        ("x=2", "x=1"),
        ("x = 1", "x=1"),
        ("x=1", "1"),
    ],
)
def test_other_answers_keep_existing_grading(answer: str, gold: str, monkeypatch: pytest.MonkeyPatch) -> None:
    """The exact-match shortcut must not introduce equation normalization."""
    seen = []

    def reject(given: str, reference: str, **kwargs) -> bool:
        seen.append((given, reference))
        return False

    monkeypatch.setattr(prime_math, "grade_answer", reject)
    monkeypatch.setattr(prime_math, "math_equal", reject)
    response = f"\\boxed{{{answer}}}"
    expected_extracted = prime_math.match_answer(response)[1]
    correct, _, extracted = prime_math.compute_score(response, gold)
    assert correct is False
    assert extracted == expected_extracted
    assert seen == [(expected_extracted, gold), (expected_extracted, gold)]


@pytest.mark.parametrize("response", [r"\boxed{x=1} then \boxed{x=2}", r"\boxed{x=1} then \fbox{x=2}"])
def test_only_last_box_is_considered(response: str) -> None:
    """A correct intermediate box must not override the final answer."""
    correct, _, _ = prime_math.compute_score(response, "x=1")
    assert correct is False


def test_last_box_can_use_either_command() -> None:
    """The final fbox must take precedence over an earlier boxed answer."""
    assert prime_math.compute_score(r"\boxed{x=2} then \fbox{x=1}", "x=1") == (True, True, "x=1")


def test_scalar_answer_keeps_existing_result() -> None:
    """Scalar rewards keep their existing extraction and grading."""
    assert prime_math.compute_score(r"The answer is $\boxed{42}$.", "42") == (True, True, "42")


@pytest.mark.parametrize("data_source", ["numina_cn_k12", "numina_aops_forum"])
def test_equation_through_default_reward_router(data_source: str) -> None:
    """The public reward router must propagate the corrected score."""
    equation = r"x - \sqrt{3}y + 2 = 0"
    assert default_compute_score(data_source, f"\\boxed{{{equation}}}", equation) == 1.0
