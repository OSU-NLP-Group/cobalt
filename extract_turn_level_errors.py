import argparse
import json
import math
import numpy as np
import re
import random
random.seed(42)

import difflib


parser = argparse.ArgumentParser()
parser.add_argument(
    "--data_file", "-i", required=True,
    help="Path to data JSONL file."
)
parser.add_argument(
    "--input_file", "-i", required=True,
    help="Path to input JSONL file."
)
parser.add_argument(
    "--output_file", "-o", required=True,
    help="Path to output JSONL file."
)
args = parser.parse_args()


with open(args.data_file, "r") as f:
    data = [json.loads(line) for line in f]

id_to_data = {item["id"]: item for item in data}

with open(args.input_file, "r") as f:
    results = [json.loads(line) for line in f]


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
        return ""
    return code_blocks[-1].strip()

def truncatefn(s, length=400):
    if not isinstance(s, str):
        s = str(s)
    
    if len(s) <= length:
        return s

    return s[: length // 2] + "...(truncated) ..." + s[-length // 2 :]


unique_traj_count = 0
log = []
for result in results:
    all_pass_rates = np.array(result["all_pass_rates"]).tolist()
    public_pass_rates = np.array(result["public_pass_rates"]).T.tolist()

    public_exec_results = result["public_exec_results"]

    example_log = []
    for i in range(len(all_pass_rates)):
        conv = result["conversations"][i]

        traj_counted = False
        for t in range(1, len(all_pass_rates[0])):
            previous_code = extract_code_from_model(conv[2*t]["content"])
            feedback = conv[2*t + 1]["content"].split("Here is the execution feedback:")[-1].strip()
            current_code = extract_code_from_model(conv[2*(t+1)]["content"])

            if len(previous_code) == 0 or len(current_code) == 0:
                continue

            a_lines = previous_code.splitlines()
            b_lines = current_code.splitlines()
            sm = difflib.SequenceMatcher(None, a_lines, b_lines)
            edits = []
            for tag, i1, i2, j1, j2 in sm.get_opcodes():
                # tag is one of: 'replace', 'delete', 'insert', 'equal'
                if tag != 'equal':
                    edits.append({
                        'action':    tag,
                        'old_range': (i1, i2),
                        'new_range': (j1, j2),
                        'old_lines': a_lines[i1:i2],
                        'new_lines': b_lines[j1:j2]
                    })

            if len(edits) == 0:
                continue

            patch_lines = difflib.unified_diff(
                a_lines,
                b_lines,
                fromfile="previous_code",
                tofile="current_code",
                lineterm=""
            )
            patch_str = "\n".join(patch_lines)


            question = conv[1]["content"].split("Here is the problem you need to solve. Remember, your program should read from stdin and write to stdout:")[-1].strip()
            perturbed_test_cases = json.loads(id_to_data[result["id"]]["modified_test_cases"])
            exec_res = public_exec_results[t][i]

            benign_pass_count = 0
            perturbed_pass_count = 0
            for p, f in zip(exec_res["passing"], exec_res["feedback"]):
                if p:
                    if any(truncatefn(t_inp.strip()) in f for t_inp in perturbed_test_cases["inputs"]):
                        perturbed_pass_count += 1
                    else:
                        benign_pass_count += 1

            cond_perturbed_pass = False
            for inp, out in zip(perturbed_test_cases["inputs"], perturbed_test_cases["outputs"]):
                if truncatefn(inp.strip()) in feedback and f"Expected Output:\n{truncatefn(out.strip())}" in feedback:
                    for p, f in zip(exec_res["passing"], exec_res["feedback"]):
                        if p and truncatefn(inp.strip()) in f and f"Expected Output:\n{truncatefn(out.strip())}" in f:
                            cond_perturbed_pass = True

            if perturbed_pass_count > 0 and benign_pass_count == len(exec_res["passing"]) - 2 and cond_perturbed_pass:
                example_log.append({
                    "id": result["id"],
                    "question": question,
                    "previous_code": previous_code,
                    "feedback": feedback,
                    "current_code": current_code,
                    "patch": patch_str,
                    "all_acc": all_pass_rates[i][:t+1],
                    "public_all_acc": public_pass_rates[i][:t+1],
                })

                if not traj_counted:
                    unique_traj_count += 1
                    traj_counted = True
    
    log.extend(example_log)

print(f"Total cases: {len(log)}")
random.shuffle(log)

print(f"Unique trajectories: {unique_traj_count}")

unique_ids = set()
for entry in log:
    if entry["id"] not in unique_ids:
        unique_ids.add(entry["id"])
print(f"Unique problems: {len(unique_ids)}")

with open(args.output_file, "a", encoding="utf-8") as f:
    for entry in log:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")