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
TASK_SEM = asyncio.Semaphore(2)
EVAL_SEM = asyncio.Semaphore(64)

ENGINE_HANDLE = serve.get_deployment_handle("VllmEngineActor", app_name="vllm_actor")
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

FEEDBACK_TEMPLATE = """Think step by step and fix any issues in your previous Python code solution to make it correct.
You should do a full analysis of the code and the feedback given below before writing the fixed code.
In your analysis, you should use the given feedback to trace the execution of the prior code and find relevant code snippets that caused the issues mentioned in the feedback.
Cite these relevant code snippets in your reasoning explicitly and explain how you would modify them to fix the issues.
If there are missing packages in the execution environment, please think of a way to implement the functionality without using those packages. You are NOT allowed to change the execution environment.
Once you have a clear idea of how to fix the code, write the complete fixed code with step-by-step comments in a single new code block. Please do NOT propose a few line changes, incomplete program outline, or partial code that requires the user to modify.

Here is the execution feedback:
{}"""

def parse_args():
    parser = argparse.ArgumentParser(
        description="Run vLLM inference on a JSONL dataset."
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
        "--temperature", type=float, default=0.6,
        help="Sampling temperature."
    )
    parser.add_argument(
        "--top_p", type=float, default=0.95,
        help="Nucleus sampling probability."
    )
    parser.add_argument(
        "--max_tokens", type=int, default=8192,
        help="Maximum number of tokens to generate per prompt."
    )
    parser.add_argument(
        "--num_completions", "-n", type=int, default=16,
        help="Number of completions to generate per prompt."
    )
    parser.add_argument(
        "--enable_thinking",
        action="store_true",
        help="Enable the model to think step by step before answering."
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

async def get_results_and_feedback(responses, test_cases):
    tasks = []
    for resp in responses:
        tasks.append(
            asyncio.create_task(
                eval_completion_wrapper(resp, test_cases)
            )
        )

    turn_exec_results = await asyncio.gather(*tasks)
    turn_pass_rates = []
    for turn_result in turn_exec_results:
        pass_rate = sum(turn_result["passing"]) / len(turn_result["passing"])
        turn_pass_rates.append(pass_rate)

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

    return turn_pass_rates, turn_feedback, turn_exec_results

async def process_one_example(example, t=8): # train t=2; eval: t=8
    prompt = example["instruction"]
    init_responses = example["responses"]
    public_test_cases = json.loads(example["public_test_cases"])

    public_pass_rates = []
    public_exec_results = []
    conversations = []
    for r in init_responses:
        if args.enable_thinking:
            if "</think>" in r:
                resp = r.split("</think>")[-1].strip()
            else:
                resp = r

            if len(resp) > 6000:
                resp = resp[:6000]
        else:
            resp = r

        conversations.append(
            [
                {"role": "system", "content": SYS},
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": resp}
            ]
        )
    
    example["raw_responses"] = [init_responses]

    turn_pass_rates, turn_feedback, turn_exec_results = await get_results_and_feedback(init_responses, public_test_cases)
    public_pass_rates.append(turn_pass_rates)
    public_exec_results.append(turn_exec_results)
    for conv, fb in zip(conversations, turn_feedback):
        conv.append({"role": "user", "content": FEEDBACK_TEMPLATE.format(fb)})

    for _ in range(t):
        gen_result = await ENGINE_HANDLE.remote(
            messages=conversations,
            temperature=args.temperature,
            top_p=args.top_p,
            max_new_tokens=args.max_tokens,
            n=1,
            is_batch=True
        )
        
        completions = gen_result["responses"]
        example["raw_responses"].append(completions)
        for conv, resp in zip(conversations, completions):
            if args.enable_thinking:
                if "</think>" in resp:
                    r = resp.split("</think>")[-1].strip()
                else:
                    r = resp
                
                if len(r) > 6000:
                    r = r[:6000]
            else:
                r = resp
            conv.append({"role": "assistant", "content": r})

        turn_pass_rates, turn_feedback, turn_exec_results = await get_results_and_feedback(completions, public_test_cases)
        public_pass_rates.append(turn_pass_rates)
        public_exec_results.append(turn_exec_results)
        for conv, fb in zip(conversations, turn_feedback):
            conv.append({"role": "user", "content": FEEDBACK_TEMPLATE.format(fb)})
            
    example["conversations"] = conversations
    example["public_pass_rates"] = public_pass_rates
    example["public_exec_results"] = public_exec_results
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