from code_reward_func import compute_score

from datasets import load_dataset
from tqdm.asyncio import tqdm

import asyncio
import json
import random

random.seed(42)

import ray
from ray import serve
EVAL_HANDLE = serve.get_deployment_handle("Evaluator", app_name="evaluator")

task_sem = asyncio.Semaphore(64)


async def eval_suite(code, test_cases):
    await asyncio.sleep(random.random())

    try:
        result = await asyncio.wait_for(
            EVAL_HANDLE.remote({"code": code, "test_cases": test_cases}),
            timeout=30,
        )
    except:
        result = {
            "passing": [False],
            "feedback": ["Execution error."]
        }
    
    return result


async def process_one_example(example, max_tests=20):
    # skip if test cases < 8 or are not well formatted
    test_cases = json.loads(example["input_output"])
    if len(test_cases["inputs"]) < 8:
        return None

    fn_name = ""
    if "fn_name" in test_cases:
        fn_name = test_cases["fn_name"]
        if example["source"] == "leetcode":
            fn_name = "Solution()." + fn_name
        # print(example)

    if not isinstance(test_cases["inputs"], list) or not isinstance(test_cases["outputs"], list):
        return None

    if len(test_cases["inputs"]) != len(test_cases["outputs"]):
        return None

    str_test_cases = {"inputs": [], "outputs": []}
    for inp, outp in zip(test_cases["inputs"], test_cases["outputs"]):
        if isinstance(inp, list):
            inp = " ".join(map(str, inp))

        if isinstance(outp, list):
            outp = " ".join(map(str, outp))
        
        str_test_cases["inputs"].append(inp)
        str_test_cases["outputs"].append(outp)

    test_cases = str_test_cases

    if any((not isinstance(inp, str) or not isinstance(outp, str)) for inp, outp in zip(test_cases["inputs"], test_cases["outputs"])):
        return None

    # no image
    if "image" in example["question"].lower() or "\n![" in example["question"]:
        return None

    # too short description
    char_len = len("".join([c for c in example["question"] if c.isalnum()]))
    if char_len < 100:
        return None

    # Sort test_cases by input length
    total_tests = len(test_cases['inputs'])
    if total_tests > max_tests:
        selected_indices = sorted(range(total_tests), key=lambda i: len(test_cases['inputs'][i]), reverse=True)

        # Select a mix of longest and shortest tests
        short_tests = max_tests // 4
        long_tests = max_tests - short_tests
        selected_indices = selected_indices[-short_tests:] + selected_indices[:long_tests]

        # Create a new dict with only the selected test cases
        selected_tests = {
            'inputs': [test_cases['inputs'][i] for i in selected_indices],
            'outputs': [test_cases['outputs'][i] for i in selected_indices]
        }
        test_cases = selected_tests

    # Check if any solution passes all test cases
    solutions = example["solutions"]
    random.shuffle(solutions)
    solutions = solutions[:16]

    for code in solutions:
        program = code.strip()

        if len(fn_name) > 0:
            program += f"\nprint({fn_name}(*input().split()))"

        results = await eval_suite(program, test_cases)
        pass_rate = sum(1 for r in results["passing"] if r) / len(results["passing"])

        if pass_rate == 1.0:
            break

    return {
        "id": example["id"],
        "difficulty": example["difficulty"],
        "question": example["question"].strip(),
        "test_cases": json.dumps(test_cases)
    }


async def process_example_w_sem(example):
    async with task_sem:
        return await process_one_example(example)


async def process_dataset():
    taco_verified = load_dataset("likaixin/TACO-verified", split="train")

    tasks = []
    for example in taco_verified:
        tasks.append(asyncio.create_task(process_example_w_sem(example)))

    results = await tqdm.gather(*tasks, total=len(tasks))
    results = [r for r in results if r is not None]

    return results


if __name__ == "__main__":
    results = asyncio.run(process_dataset())
    print(f"Processed {len(results)} examples")
    with open("verl/data/taco_verified_filtered.jsonl", "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")
