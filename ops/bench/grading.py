"""Grading a free-text answer against a gold answer.

Exact string matching fails on "Jane Ballou" vs "Her name would be Jane
Ballou." So, like the FRAMES and BrowseComp papers, a model grades: it
pulls the final answer out of the response and says whether it means the
same as the gold answer. The prompt follows the grader OpenAI published
for BrowseComp (adapted from Humanity's Last Exam): judge only the final
answer, allow small numeric differences, and don't re-solve the problem.

The grader sees the gold answer; sora never does.
"""

from __future__ import annotations

from loop.llm import ask_json, schema

GRADE_FORMAT = schema({
    "extracted_final_answer": {"type": "string"},
    "reasoning": {"type": "string"},
    "correct": {"type": "boolean"},
})

GRADER_PROMPT = """Judge whether the response to the question below gives the correct final answer.

[question]: {question}

[response]: {response}

[correct_answer]: {gold}

extracted_final_answer: the final exact answer taken from the response. Write "None" if
the response has no final answer.
reasoning: explain only whether extracted_final_answer matches [correct_answer]. Don't
comment on the background of the problem, don't solve it again, and don't argue for any
answer other than [correct_answer]. Focus on meaningful differences.
correct: true if extracted_final_answer matches [correct_answer], or is within a small
margin of error for numerical answers (rounding, units written differently). False if
there is any inconsistency, ambiguity, a non-equivalent answer, or several answers
offered without committing to one."""


def grade_answer(provider, model: str, question: str, response: str, gold: str, notify=None) -> dict:
    """{"checks": {"correct": bool}, "judge": {...}} in the shape run_case expects."""
    if not (response or "").strip():
        return {"checks": {"correct": False},
                "judge": {"score": 0.0, "grade": 1, "reason": "no answer", "extracted": "None"}}
    answer = ask_json(provider, model, GRADER_PROMPT.format(question=question[:6000], response=response[:8000],
                                                            gold=gold), GRADE_FORMAT, notify, name="grader",
                      max_tokens=2048)
    ok = bool(answer["correct"])
    return {"checks": {"correct": ok},
            "judge": {"score": 1.0 if ok else 0.0, "grade": 5 if ok else 1,
                      "reason": f"answered {answer['extracted_final_answer']!r}; gold {gold!r}. {answer['reasoning']}",
                      "extracted": answer["extracted_final_answer"]}}
