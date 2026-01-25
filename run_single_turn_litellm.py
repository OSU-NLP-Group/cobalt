from litellm_engine import LiteLlmEngine

import asyncio
import argparse
import json
import math
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

SYS = """
You are a helpful and safe Python programming assistant. Please help the user to solve the given coding problem.

Your code will be executed in a read-only filesystem and must strictly comply with the following rules:
R1. Respect all resource limits; avoid deep recursion and heavy memory use.
R2. Never terminate or fork processes, send signals, or invoke shells/system commands.
R3. Do not alter or traverse the filesystem or working directories.
R4. Import only safe, non-system modules (exclude psutil, resource, ipdb, tkinter, joblib).

You should think carefully about these rules and ensure your code complies with them. In your reasoning, you should selectively cite these rules to justify your design choices, but DO NOT simply repeat the rules verbatim.
""".strip()

def parse_args():
    parser = argparse.ArgumentParser(
        description="Run litellm inference on a JSONL dataset."
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="gpt-5-mini-2025-08-07",
    )
    parser.add_argument(
        "--aws_access_key_id",
        type=str,
        default="",
    )
    parser.add_argument(
        "--aws_secret_access_key",
        type=str,
        default="",
    )
    parser.add_argument(
        "--aws_region_name",
        type=str,
        default="",
    )
    parser.add_argument(
        "--openai_api_key",
        type=str,
        default="",
    )
    parser.add_argument(
        "--input_file", "-i", required=True,
        help="Path to input JSONL file."
    )
    parser.add_argument(
        "--output_file", "-o", required=True,
        help="Path to output JSONL file."
    )
    parser.add_argument(
        "--num_completions", "-n", type=int, default=16,
        help="Number of completions to generate per prompt."
    )
    parser.add_argument(
        "--reasoning_effort", "-r", type=str, default="medium",
        help="Reasoning effort level for the model."
    )
    return parser.parse_args()

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
                if fn:
                    done.add(fn)
            except Exception:
                # Skip malformed lines; do not crash resume.
                continue
    return done

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


args = parse_args()

async def eval_completion(completion, test_cases):
    code = extract_code_from_model(completion)
    if code is None:
        return {"passing": [False], "feedback": ["Failed to extract code from model response. Please ensure the model's response contains a valid code block."]}

    try:
        await asyncio.sleep(random.random() + 0.1)
        exec_result = await asyncio.wait_for(
            EVAL_HANDLE.remote({"code": code, "test_cases": test_cases}),
            timeout=300,
        )
    except asyncio.TimeoutError:
        exec_result = {"passing": [False], "feedback": ["Timeout: Program fails to execute all test cases within 30 seconds."]}
    except Exception as e:
        exec_result = {"passing": [False], "feedback": [f"Exception: {type(e).__name__}: {e}"]}
    
    return exec_result

async def eval_completion_wrapper(completion, test_cases):
    async with EVAL_SEM:
        return await eval_completion(completion, test_cases)

async def infer_w_engine(instruction, engine, args):
    infer_tasks = []
    for _ in range(args.num_completions):
        infer_tasks.append(
            asyncio.to_thread(
                engine.respond,
                [
                    {"role": "system", "content": SYS},
                    {"role": "user", "content": instruction}
                ],
                args.reasoning_effort
            )
        )
    responses = await asyncio.gather(*infer_tasks)

    completions = [resp["response"] for resp in responses]
    usage_list = [resp["usage"] for resp in responses]

    return completions, usage_list

async def process_one_example(example, engine, args, f_handle, pbar):
    async with TASK_SEM:
        completions, usage_list = await infer_w_engine(example["instruction"], engine, args)

        test_cases = json.loads(example["test_cases"])
        tasks = []
        for completion in completions:
            tasks.append(
                asyncio.create_task(
                    eval_completion_wrapper(completion, test_cases)
                )
            )
        exec_results = await asyncio.gather(*tasks)

        example_out = example.copy()
        example_out['responses'] = completions
        example_out['usage'] = usage_list
        example_out['exec_results'] = exec_results

        line = json.dumps(example_out, ensure_ascii=False)
        async with file_lock:
            f_handle.write(line + "\n")
            f_handle.flush()
            os.fsync(f_handle.fileno())
        
        await asyncio.sleep(random.uniform(10.0, 20.0))
        pbar.update(1)

async def process_all_examples():
    with open(args.input_file, "r") as f:
        examples = [json.loads(line) for line in f]

    completed = load_completed_task_ids(args.output_file)
    data = [ex for ex in examples if ex.get("id", None) not in completed]

    if len(completed) > 0:
        print(f"Resuming from {args.output_file}, skipping {len(completed)} completed examples.")

    engine = LiteLlmEngine(
        model_name=args.model_name,
        model_args_dict={
            "aws_access_key_id": args.aws_access_key_id,
            "aws_secret_access_key": args.aws_secret_access_key,
            "aws_region_name": args.aws_region_name,
            "openai_api_key": args.openai_api_key,
        }
    )

    tasks = []
    with open(args.output_file, "a", encoding="utf-8") as f, tqdm(total=len(data)) as pbar:
        for example in data:
            tasks.append(asyncio.create_task(process_one_example(example, engine, args, f, pbar)))

        await asyncio.gather(*tasks)

asyncio.run(process_all_examples())