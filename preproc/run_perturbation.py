import argparse
import json
import random
from tqdm import tqdm

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
    return parser.parse_args()

args = parse_args()

with open(args.input_file, "r") as f:
    original_data = [json.loads(line) for line in f]

adversarial_data = []
for data in tqdm(original_data):
    public_test_cases = json.loads(data["public_test_cases"])
    public_io_pairs = [(i, o) for i, o in zip(public_test_cases["inputs"], public_test_cases["outputs"])]

    random.shuffle(public_io_pairs)
    unique_outputs = set()
    perturbed_io_pairs = []
    original_io_pairs = []
    for inp, output in public_io_pairs:
        if output.strip() not in unique_outputs:
            unique_outputs.add(output.strip())
            perturbed_io_pairs.append((inp, output))
        else:
            original_io_pairs.append((inp, output))

    if len(perturbed_io_pairs) < 2:
        continue

    while perturbed_io_pairs[0][1] == perturbed_io_pairs[1][1]:
        random.shuffle(perturbed_io_pairs)

    final_io_pairs = [(perturbed_io_pairs[0][0], perturbed_io_pairs[1][1]), (perturbed_io_pairs[1][0], perturbed_io_pairs[0][1])]
    data["modified_test_cases"] = json.dumps({
        "inputs": [io[0] for io in final_io_pairs],
        "outputs": [io[1] for io in final_io_pairs],
    })

    final_io_pairs = final_io_pairs + perturbed_io_pairs[2:] + original_io_pairs
    data["perturbed_public_test_cases"] = json.dumps({
        "inputs": [io[0] for io in final_io_pairs],
        "outputs": [io[1] for io in final_io_pairs],
    })
    adversarial_data.append(data)

print(len(adversarial_data))

with open(args.output_file, "w") as f:
    for data in adversarial_data:
        f.write(json.dumps(data) + "\n")