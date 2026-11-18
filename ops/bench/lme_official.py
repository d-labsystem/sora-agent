"""LongMemEval's own grader, unchanged, so sora's scores sit beside published ones.

Memory systems publish LongMemEval scores graded by the paper's script
(src/evaluation/evaluate_qa.py) with GPT-4o. Sora's usual judge (ops/bench/longmemeval.py,
"lme-v1") is adapted from it: structured output, Claude Haiku, reworded rules. Close
isn't comparable, so this module is the original, copied verbatim from
https://github.com/xiaowu0162/LongMemEval at commit d6dc8b50a2d9 (2024-11-24):

    get_anscheck_prompt   the five prompt templates, word for word
    the call              gpt-4o-2024-08-06, temperature 0, max_tokens 10, one user message
    the verdict           'yes' in the reply, lowercased

Abstention is decided as theirs is: '_abs' in the question id. The reply isn't truncated.
Needs OPENAI_API_KEY. ~$0.002 per answer.
"""

from __future__ import annotations

import time

MODEL = "gpt-4o-2024-08-06"
PRICE = (2.50, 10.00)          # $ per million input / output tokens
SOURCE = "LongMemEval src/evaluation/evaluate_qa.py @ d6dc8b50a2d9"


# ---- verbatim from evaluate_qa.py (only the indentation and this comment are sora's)

def get_anscheck_prompt(task, question, answer, response, abstention=False):
    if not abstention:
        if task in ['single-session-user', 'single-session-assistant', 'multi-session']:
            template = "I will give you a question, a correct answer, and a response from a model. Please answer yes if the response contains the correct answer. Otherwise, answer no. If the response is equivalent to the correct answer or contains all the intermediate steps to get the correct answer, you should also answer yes. If the response only contains a subset of the information required by the answer, answer no. \n\nQuestion: {}\n\nCorrect Answer: {}\n\nModel Response: {}\n\nIs the model response correct? Answer yes or no only."
            prompt = template.format(question, answer, response)
        elif task == 'temporal-reasoning':
            template = "I will give you a question, a correct answer, and a response from a model. Please answer yes if the response contains the correct answer. Otherwise, answer no. If the response is equivalent to the correct answer or contains all the intermediate steps to get the correct answer, you should also answer yes. If the response only contains a subset of the information required by the answer, answer no. In addition, do not penalize off-by-one errors for the number of days. If the question asks for the number of days/weeks/months, etc., and the model makes off-by-one errors (e.g., predicting 19 days when the answer is 18), the model's response is still correct. \n\nQuestion: {}\n\nCorrect Answer: {}\n\nModel Response: {}\n\nIs the model response correct? Answer yes or no only."
            prompt = template.format(question, answer, response)
        elif task == 'knowledge-update':
            template = "I will give you a question, a correct answer, and a response from a model. Please answer yes if the response contains the correct answer. Otherwise, answer no. If the response contains some previous information along with an updated answer, the response should be considered as correct as long as the updated answer is the required answer.\n\nQuestion: {}\n\nCorrect Answer: {}\n\nModel Response: {}\n\nIs the model response correct? Answer yes or no only."
            prompt = template.format(question, answer, response)
        elif task == 'single-session-preference':
            template = "I will give you a question, a rubric for desired personalized response, and a response from a model. Please answer yes if the response satisfies the desired response. Otherwise, answer no. The model does not need to reflect all the points in the rubric. The response is correct as long as it recalls and utilizes the user's personal information correctly.\n\nQuestion: {}\n\nRubric: {}\n\nModel Response: {}\n\nIs the model response correct? Answer yes or no only."
            prompt = template.format(question, answer, response)
        else:
            raise NotImplementedError
    else:
        template = "I will give you an unanswerable question, an explanation, and a response from a model. Please answer yes if the model correctly identifies the question as unanswerable. The model could say that the information is incomplete, or some other information is given but the asked information is not.\n\nQuestion: {}\n\nExplanation: {}\n\nModel Response: {}\n\nDoes the model correctly identify the question as unanswerable? Answer yes or no only."
        prompt = template.format(question, answer, response)
    return prompt


# ---- the call, as evaluate_qa.py makes it

_client = None


def client():
    global _client
    if _client is None:
        from openai import OpenAI
        _client = OpenAI()                       # OPENAI_API_KEY
    return _client


def grade(question_id: str, task: str, question: str, answer: str, response: str, openai=None,
          retries: int = 5) -> dict:
    """{correct, said, cost_usd}: GPT-4o's verdict on one answer, exactly as the paper's script asks."""
    prompt = get_anscheck_prompt(task, question, answer, response, abstention='_abs' in question_id)
    api = openai or client()
    for attempt in range(retries):               # their script backs off too (chat_completions_with_backoff)
        try:
            completion = api.chat.completions.create(model=MODEL, messages=[{"role": "user", "content": prompt}],
                                                     n=1, temperature=0, max_tokens=10)
            break
        except Exception:
            if attempt == retries - 1:
                raise
            time.sleep(2 ** attempt)
    said = completion.choices[0].message.content.strip()
    usage = completion.usage
    cost = (usage.prompt_tokens * PRICE[0] + usage.completion_tokens * PRICE[1]) / 1e6 if usage else 0.0
    return {"correct": 'yes' in said.lower(), "said": said, "cost_usd": cost}
