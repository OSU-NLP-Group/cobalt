import argparse
import json
import numpy as np
import random
random.seed(42)

from transformers import AutoTokenizer


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
        "--model_name", type=str, default="Qwen/Qwen3-8B",
        help="Model name."
    )
    parser.add_argument(
        "--is_test", action="store_true",
        help="Whether the input file is a test set."
    )
    return parser.parse_args()

args = parse_args()

tokenizer = AutoTokenizer.from_pretrained(args.model_name)
num_turns = 3
is_test = args.is_test
with open(args.input_file, "r") as f:
    data = [json.loads(line) for line in f]

skipped_num_examples = 0
unique_num_examples = 0
public_test_passes = 0

conversation_data = []
for example in data:
    turn_0_exec_res = example["exec_results"]

    if all(all(result["passing"]) for result in turn_0_exec_res):
        skipped_num_examples += 1
        continue

    # Turn 0 is the same for all conversations
    conversation_data.append({
        "example_id": example["id"],
        "conversation": example["conversations"][0][:2],
        "test_cases": example["test_cases"],
        "difficulty": example["difficulty"],
        "prev_best_pass_rate": 0.0,
    })

    num_test_cases = len(json.loads(example["test_cases"])["inputs"])

    all_pass_rates = (np.array(example["all_pass_rates"]) * (num_test_cases - 4) + np.array(example["public_pass_rates"]).T * 4) / num_test_cases
    max_pass_rates = np.max(all_pass_rates, axis=1)
    min_pass_rates = np.min(all_pass_rates, axis=1)
    pass_rate_diffs = (max_pass_rates - min_pass_rates).tolist()

    m = 4
    n = 16

    if is_test:
        indexed_pass_rates = [
            (i, pr) 
            for i, pr in enumerate(pass_rate_diffs)
            if pr > 0
        ]
    else:
        indexed_pass_rates = [
            (i, pr) 
            for i, pr in enumerate(pass_rate_diffs)
            if pr > 0.0 and max_pass_rates[i] == 1.0
        ]
    sorted_indicies = [idx for idx, _ in sorted(indexed_pass_rates, key=lambda x: x[1])]

    if is_test and len(indexed_pass_rates) > 0:
        min_index = sorted_indicies[0]
        subset = [min_index]
        conversations = [example["conversations"][min_index][:8]]
    else:
        if len(indexed_pass_rates) > m:
            # https://arxiv.org/pdf/2504.13818
            n = min(n, len(indexed_pass_rates))
            subset = sorted_indicies[:m]
            for k in range(m):
                new_subset = subset[:m - k] + sorted_indicies[n - k:]
                if np.var([pass_rate_diffs[i] for i in new_subset]) > np.var([pass_rate_diffs[i] for i in subset]):
                    subset = new_subset
            conversations = [example["conversations"][i] for i in subset]
        else:
            conversations = [example["conversations"][i] for i, pr in indexed_pass_rates]
            subset = [i for i, pr in indexed_pass_rates]

    if len(conversations) > 0:
        unique_num_examples += 1

    for conv, idx in zip(conversations, subset):
        for turn in range(4, 4 + num_turns * 2, 2):
            if "The code has passed all public test cases" in conv[turn - 1]["content"] and "The code has passed all public test cases" in conv[turn - 3]["content"]:
                break

            if len(tokenizer.apply_chat_template(conv[:turn], add_generation_prompt=True)) > 10000:
                break

            if turn // 2 < num_turns and all_pass_rates[idx, turn // 2 - 2] == 1.0 and all_pass_rates[idx, turn // 2] == 1.0:
                public_test_passes += 1
                break

            prev_pass_rates = all_pass_rates[idx][:(turn - 2) // 2].tolist()

            conversation_data.append({
                "example_id": example["id"],
                "conversation": conv[:turn],
                "test_cases": example["test_cases"],
                "difficulty": example["difficulty"],
                "prev_best_pass_rate": max(prev_pass_rates) if prev_pass_rates else 0.0
            })

print(f"Skipped examples: {skipped_num_examples}")
print(f"Unique examples with conversations extracted: {unique_num_examples}")
print(f"Total conversations extracted: {len(conversation_data)}")
print(f"Total turns where public test cases passed: {public_test_passes}")

random.shuffle(conversation_data)

with open(args.output_file, "w") as f:
    for item in conversation_data:
        f.write(json.dumps(item) + "\n")
