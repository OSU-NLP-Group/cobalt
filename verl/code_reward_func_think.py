import asyncio
import json
import math
import random
import re

import ray
from ray import serve

EVAL_HANDLE = serve.get_deployment_handle("Evaluator", app_name="evaluator")

import tiktoken
enc = tiktoken.get_encoding("o200k_base")


KEYWORDS = ["wait", "mistake", "however", "but", "retry", "error", "verify", "wrong", "evaluate"]
# Pre-compile regex patterns for performance
_patterns = [(kw, re.compile(rf"\b{re.escape(kw)}\b")) for kw in KEYWORDS]


def detect_keywords(solution_str: str):
    """
    Count occurrences of each KEYWORD in the solution string.
    Returns a dictionary {keyword: count} for keywords found at least once.
    """
    counts = {}
    for line in solution_str.splitlines():
        for kw, pat in _patterns:
            if pat.search(line):
                counts[kw] = counts.get(kw, 0) + len(pat.findall(line))
    # Remove keywords with zero count
    return {kw: count for kw, count in counts.items() if count > 0}


def extract_code_from_model(model_response: str):
    """
    Extracts the code from a Markdown-style code block in an LLM output.

    Args:
        model_response (str): The text output from the LLM.

    Returns:
        str: The extracted code, or an empty string if no code block is found.
    """
    code_blocks = re.findall(r"```(?:\w+)?\n(.*?)```", model_response, re.DOTALL)
    if not code_blocks:
        return None
    return code_blocks[-1].strip()


def find_repetitive_pattern(s: str, min_len: int = 32, min_repeats: int = 8):
    """
    Returns (found, block_len, repeats, span) where:
      - found: whether a pattern was detected
      - block_len: the length of the repeating block that triggered the detection
      - repeats: how many consecutive repeats were found (>= min_repeats)
      - span: (start, end) indices of the matched region
    """
    n = len(s)
    max_len = n // min_repeats
    for L in range(min_len, max_len + 1):
        # DOTALL so '.' matches newlines
        m = re.search(rf"(.{{{L}}})\1{{{min_repeats-1},}}", s, flags=re.DOTALL)
        if m:
            total = m.end() - m.start()
            repeats = total // L
            return True, L, repeats, m.span()
    return False, None, None, None


def has_repetitive_pattern(s: str, min_len=32, min_repeats=8):
    return find_repetitive_pattern(s, min_len, min_repeats)[0]


def format_reward(solution_str: str, correctness_reward: float) -> float:
    format_penalty = -0.9

    ### Repetitive pattern check
    if has_repetitive_pattern(solution_str):
        return format_penalty
    
    ### Think block format check
    think_end = solution_str.find("</think>")
    if think_end == -1:
        return format_penalty

    if solution_str[think_end:].count("</think>") != 1:
        return format_penalty
    
    ### Code block format check
    response = solution_str[think_end + 8:].strip()
    code_start = response.find("```python")
    if code_start == -1:
        return format_penalty

    if response.count("```python") != 1:
        return format_penalty

    code_end = response.find("\n```", code_start + 9)
    if code_end == -1:
        return format_penalty

    if response[code_start + 9:].count("\n```") != 1:
        return format_penalty

    if ("`" in response[code_start + 9:code_end]):
        return format_penalty

    ### Thinking quality
    think_part = solution_str[:think_end].strip()
    think_len = len(enc.encode(think_part))
    response_len = len(enc.encode(response))
    explanation_len = len(enc.encode(response[code_end + 4:].strip()))

    if correctness_reward >= 0.0:
        if think_len <= 4096 and response_len <= 2048 and explanation_len <= 512:
            return 0.0
        else:
            return -0.01  # slight penalty for overly long correct answers
    
    # Smoothly scale the reward based on the length of the think part
    format_reward = 0.0
    if think_len <= 4096:
        format_reward = math.tanh(min(think_len, 4096) / 1024) * 0.075  # up to +0.075

        # Bonus for metacognitive keywords in think part
        keyword_bonus = 0.0
        keyword_count = detect_keywords(think_part.lower())
        for kw, count in keyword_count.items():
            if count <= 4:
                keyword_bonus += 0.005
        format_reward += min(keyword_bonus, 0.025)  # cap at +0.025
    else:
        format_reward = -0.1

    # Penalty for long responses
    if response_len > 2048:
        format_reward -= 0.1

    # Penalty for long explanations
    if explanation_len > 512:
        format_reward -= 0.1

    return format_reward


async def compute_score(
    data_source,
    solution_str,
    ground_truth,
    extra_info=None,
    sandbox_fusion_url=None,
    concurrent_semaphore=None,
    memory_limit_mb=None,
):
    code = extract_code_from_model(solution_str)
    if code is None:
        if extra_info is not None and extra_info.get("is_val", False):
            return 0.0
        else:
            return -1.0

    test_cases = json.loads(ground_truth)
    try:
        await asyncio.sleep(random.random() * 2 + 0.1)
        exec_result = await asyncio.wait_for(
            EVAL_HANDLE.remote({"code": code, "test_cases": test_cases}),
            timeout=30,
        )
    except asyncio.TimeoutError:
        exec_result = {"passing": [False], "feedback": ["Timeout: Program fails to execute all test cases within 30 seconds."]}
    except Exception as e:
        exec_result = {"passing": [False], "feedback": [f"Exception: {type(e).__name__}: {e}"]}
    
    # Correctness
    pass_rate = sum(1 for p in exec_result["passing"] if p) / len(exec_result["passing"])
    prev_best_pass_rate = extra_info.get("prev_best_pass_rate", 0.0) if extra_info is not None else 0.0

    if pass_rate == 1.0:
        reward_value = 1.0
    elif pass_rate >= 0.5:
        reward_value = 0.2
    elif pass_rate > 0.0:
        reward_value = 0.0
    else:
        reward_value = -0.1

    # Improvement
    if pass_rate < 1.0 and prev_best_pass_rate > 0.0:
        if pass_rate - prev_best_pass_rate > 0.0:
            reward_value += (pass_rate - prev_best_pass_rate) / (1.0 - prev_best_pass_rate) * 0.1
        else:
            # Penalize regression with a negative reward
            reward_value = -0.1 + (pass_rate - prev_best_pass_rate) / prev_best_pass_rate * 0.1

    # Format reward
    reward_value += format_reward(solution_str, reward_value)

    if extra_info is not None and extra_info.get("is_val", False):
        return (1.0 if pass_rate == 1.0 else 0.0)  # 1 for all correct, 0 otherwise
    else:
        # Cap the reward to be within [-1, 1]
        reward_value = max(-1.0, min(1.0, reward_value))
        return reward_value