import asyncio
import argparse
import json
import math
import numpy as np
import os
import random
import re
from tqdm import tqdm

import ray
from ray import serve

file_lock = asyncio.Lock()
TASK_SEM = asyncio.Semaphore(4)
EVAL_SEM = asyncio.Semaphore(64)

EVAL_HANDLE = serve.get_deployment_handle("Evaluator", app_name="evaluator")


parser = argparse.ArgumentParser()
parser.add_argument(
    "--input_file", "-i", required=True,
    help="Path to input JSONL file."
)
parser.add_argument(
    "--output_file", "-o", required=True,
    help="Path to output JSONL file."
)
args = parser.parse_args()


def pass_at_k(n: int, c: int, k: int) -> float:
    """
    Compute pass@k for a single problem.
    
    Args:
      n: total number of samples generated for this problem
      c: number of those samples that are correct (c ≥ 0)
      k: number of samples you’re “allowed” to pick
      
    Returns:
      Probability that at least one of k randomly chosen samples is correct.
    """
    if c == 0:
        return 0.0
    if k > n:
        raise ValueError(f"k={k} must be ≤ n={n}")
    # If there are fewer incorrect samples than k, you're guaranteed to pick at least one correct
    if n - c < k:
        return 1.0
    # exact combinatorial formula
    return 1.0 - math.comb(n - c, k) / math.comb(n, k)

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

def load_completed_task_ids(path: str) -> set:
    """
    Read an existing JSONL file and return the set of `id` values
    already completed. Robust to blank/corrupt lines.
    """
    done = set()
    if not os.path.exists(path):
        return done
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                fn = obj.get("id", None)
                has_pass_rate = obj.get("all_pass_rates", None) is not None
                # has_pass_rate = obj.get("public_pass_rates", None) is not None
                if fn and has_pass_rate:
                    done.add(fn)
            except Exception:
                # Skip malformed lines; do not crash resume.
                continue
    return done

async def eval_completion(completion, test_cases):
    code = extract_code_from_model(completion)
    if code is None:
        return {"passing": [False], "feedback": ["Failed to extract code from model response. Please ensure the model's response contains a valid code block."]}

    try:
        await asyncio.sleep(random.random() + 0.1)
        exec_result = await asyncio.wait_for(
            EVAL_HANDLE.remote({"code": code, "test_cases": test_cases}),
            timeout=300.0,
        )
    except asyncio.TimeoutError:
        print("Timeout during evaluation.", flush=True)
        exec_result = {"passing": [False], "feedback": ["Timeout: Program fails to execute all test cases within 30 seconds."]}
    except Exception as e:
        print(f"Exception during evaluation: {e}", flush=True)
        exec_result = {"passing": [False], "feedback": [f"Exception: {type(e).__name__}: {e}"]}
    
    return exec_result

async def eval_completion_wrapper(completion, test_cases):
    async with EVAL_SEM:
        return await eval_completion(completion, test_cases)

async def get_results_and_feedback(responses, test_cases):
    tasks = []
    for resp in responses:
        tasks.append(
            asyncio.create_task(
                eval_completion_wrapper(resp, test_cases)
            )
        )

    turn_exec_results = await asyncio.gather(*tasks)
    turn_feedback = []
    for turn_result in turn_exec_results:
        feedback_str = ""
        for passing, feedback in zip(turn_result["passing"], turn_result["feedback"]):
            if not passing:
                feedback_str = feedback
                break

        if feedback_str == "":
            feedback_str = "The code has passed all public test cases. Please try your best to think about possible edge cases and fix any remaining issues in the code to pass the hidden test cases."

        turn_feedback.append(feedback_str)

    return turn_feedback

async def process_one_example(example):
    conversations = example["conversations"]
    hidden_test_cases = json.loads(example["hidden_test_cases"])

    all_pass_rates = []
    for conv in conversations:
        tasks = []
        for i in range(2, len(conv), 2):
            tasks.append(asyncio.create_task(
                eval_completion_wrapper(conv[i]["content"], hidden_test_cases)
            ))

        exec_results = await asyncio.gather(*tasks)
        turn_pass_rates = [
            sum([int(p) for p in turn_result["passing"]]) / len(turn_result["passing"]) 
            for turn_result in exec_results
        ]
        all_pass_rates.append(turn_pass_rates)

    example["all_pass_rates"] = all_pass_rates
    return example

async def process_one_example_wrapper(example, f_handle, pbar):
    async with TASK_SEM:
        result = await process_one_example(example)

        line = json.dumps(result, ensure_ascii=False)
        async with file_lock:
            f_handle.write(line + "\n")
            f_handle.flush()
            os.fsync(f_handle.fileno())

        pbar.update(1)

async def process_all_examples():
    with open(args.input_file, "r") as f:
        examples = [json.loads(line) for line in f]

    completed = load_completed_task_ids(args.output_file)
    data = [ex for ex in examples if ex.get("id", None) not in completed]

    if len(completed) > 0:
        print(f"Resuming from {args.output_file}, skipping {len(completed)} completed examples.")

    tasks = []
    with open(args.output_file, "a", encoding="utf-8") as f, tqdm(total=len(data)) as pbar:
        for example in data:
            tasks.append(asyncio.create_task(process_one_example_wrapper(example, f, pbar)))

        await asyncio.gather(*tasks)

asyncio.run(process_all_examples())

with open(args.output_file, "r") as f:
    results = [json.loads(line) for line in f]

total_pass_rates = np.zeros(9)
total_acc_at_1 = np.zeros(9)

for result in results:
    all_pass_rates = np.array(result["all_pass_rates"])
    total_pass_rates += np.mean(all_pass_rates, axis=0)

    # Count successful turns per position (shape will be (9,))
    all_acc = np.where(all_pass_rates == 1.0, 1, 0).sum(axis=0)

    # Compute pass@1 elementwise using integer counts
    all_acc_at_1 = np.array([pass_at_k(16, int(c), 1) for c in all_acc])
    total_acc_at_1 += all_acc_at_1

total_pass_rates /= len(results)
total_acc_at_1 /= len(results)

print("Hidden Test Cases Pass @ 1:", (total_acc_at_1 * 100).round(2).tolist())
print("Relative Changes:", ((total_acc_at_1 - total_acc_at_1[0]) / total_acc_at_1[0] * 100).round(2).tolist())