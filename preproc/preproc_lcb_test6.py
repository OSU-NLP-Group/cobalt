from datasets import load_dataset
from tqdm import tqdm

import json
import os

import zlib
import pickle
import base64

INST = """Please first think step by step and then write a complete Python program that solves the given problem.
You should carefully analyze the problem, creatively brainstorm different approaches that comply with the rules, actively reflect on your current approach to correct identified mistakes, proactively try different approaches, and make further improvements until you are confident in your solution.
Please explicitly write out all your thoughts, reasoning, and reflections in detail to help the user understand your thought process.
After wrapping up your thinking, you should implement your program. Your program should be formatted in a single code block and include detailed explanations for each step in the comments. For example:
```python
# Print the sum of two numbers 
def add_numbers(a, b):
    # 1) Calculate the sum of a and b
    c = a + b
    # 2) Return the result
    return c
```

Here is the problem you need to solve. Remember, your program should read from stdin and write to stdout:
{}"""

with open("test6.jsonl", "r") as f:
    lcb = [json.loads(line) for line in f]

data = []
for example in tqdm(lcb):
    public_test_cases = json.loads(example["public_test_cases"])
    p_test_cases = {
        "inputs": [],
        "outputs": []
    }
    for tc in public_test_cases:
        p_test_cases["inputs"].append(tc["input"])
        p_test_cases["outputs"].append(tc["output"])
    example["public_test_cases"] = json.dumps(p_test_cases)

    private_test_cases = json.loads(
        pickle.loads(
            zlib.decompress(
                base64.b64decode(example["private_test_cases"].encode("utf-8"))  # type: ignore
            )
        )
    )

    h_test_cases = {
        "inputs": [],
        "outputs": []
    }
    for tc in private_test_cases:
        h_test_cases["inputs"].append(tc["input"])
        h_test_cases["outputs"].append(tc["output"])
    
    example["hidden_test_cases"] = json.dumps(h_test_cases)
    example["test_cases"] = json.dumps({
        "inputs": p_test_cases["inputs"] + h_test_cases["inputs"],
        "outputs": p_test_cases["outputs"] + h_test_cases["outputs"],
    })
    example.pop("private_test_cases")

    raw_question = example.pop("question_content")
    if "\nExample 1:\n\n" in raw_question:
        raw_question = raw_question.split("\nExample 1:\n\n")[0]
    elif "\nSample Input 1\n\n" in raw_question:
        raw_question = raw_question.split("\nSample Input 1\n\n")[0]

    example["instruction"] = INST.format(raw_question.strip())
    example["id"] = example.pop("question_id")

    data.append(example)

with open("data/lcb_v6.jsonl", "w") as f:
    for example in data:
        f.write(json.dumps(example) + "\n")
