import argparse
import os
import re
import datasets

SYS = """
You are a helpful and safe Python programming assistant. Please help the user to solve the given coding problem.

Your code will be executed in a read-only filesystem and must strictly comply with the following rules:
R1. Respect all resource limits; avoid deep recursion and heavy memory use.
R2. Never terminate or fork processes, send signals, or invoke shells/system commands.
R3. Do not alter or traverse the filesystem or working directories.
R4. Import only safe, non-system modules (exclude psutil, resource, ipdb, tkinter, joblib).

You should think carefully about these rules and ensure your code complies with them. In your reasoning, you should selectively cite these rules to justify your design choices, but DO NOT simply repeat the rules verbatim.
""".strip()

def process_dataset(fname, split):
    data = datasets.load_dataset("json", data_files=fname)["train"]

    def make_map_fn(split):
        def process_fn(example):
            instruction = example["instruction"]
            test_cases = example["test_cases"]

            data = {
                "data_source": "code_taco_verified",
                "prompt": [
                    {"role": "system", "content": SYS},
                    {"role": "user", "content": instruction}
                ],
                "ability": "code",
                "reward_model": {"style": "rule", "ground_truth": test_cases},
                "extra_info": {
                    "split": split,
                    "index": example.get("id", -1),
                    "instruction": instruction,
                },
            }
            return data

        return process_fn

    verl_dataset = data.map(function=make_map_fn(split))
    return verl_dataset


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_dir", default=None, help="The save directory for the preprocessed dataset.")
    parser.add_argument("--train_data_file", default="data/taco_verified_filtered_train.jsonl", help="Path to the training data file.")
    parser.add_argument("--val_data_file", default="data/taco_verified_filtered_dev.jsonl", help="Path to the validation data file.")
    args = parser.parse_args()

    train_dataset = process_dataset(args.train_data_file, "train")
    test_dataset = process_dataset(args.val_data_file, "test")
    
    local_save_dir = args.local_dir
    train_dataset.to_parquet(os.path.join(local_save_dir, "train.parquet"))
    test_dataset.to_parquet(os.path.join(local_save_dir, "test.parquet"))