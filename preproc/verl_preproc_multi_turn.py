import argparse
import os
import re
import datasets

def process_dataset(fname, split):
    data = datasets.load_dataset("json", data_files=fname)["train"]

    def make_map_fn(split):
        def process_fn(example):
            test_cases = example["test_cases"]

            data = {
                "data_source": "code_taco_verified",
                "prompt": example["conversation"],
                "ability": "code",
                "reward_model": {"style": "rule", "ground_truth": test_cases},
                "extra_info": {
                    "split": split,
                    "index": example.get("example_id", -1),
                    "conversation": example["conversation"],
                    "prev_best_pass_rate": example.get("prev_best_pass_rate", 0.0),
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