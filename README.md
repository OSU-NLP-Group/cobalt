# Cobalt

Code and data for the paper "[Bridging Online and Offline RL: Contextual Bandit Learning for Multi-Turn Code Generation](#)".


## Updates:
- 2026/01/29: Initial release.

## Table of Contents
- [Summary](#summary)
- [Setup](#setup)
- [Datasets and Preprocessing](#datasets-and-preprocessing)
- [Training](#training)
- [Inference](#inference)
- [Evaluation](#evaluation)
- [Citation](#citation) 

## Summary

Recently, there have been significant research interests in training large language models (LLMs) with reinforcement learning (RL) on real-world tasks, such as multi-turn code generation. While online RL tends to perform better than offline RL, its higher training cost and instability hinders wide adoption. In this paper, we build on the observation that multi-turn code generation can be formulated as a one-step recoverable Markov decision process and propose contextual bandit learning with offline trajectories (Cobalt), a new method that combines the benefits of online and offline RL. Cobalt first collects code generation trajectories using a reference LLM and divides them into partial trajectories as contextual prompts. Then, during online bandit learning, the LLM is trained to complete each partial trajectory prompt through single-step code generation. Cobalt substantially outperforms two online RL baselines, GRPO and VeRPO, and improves R1-Distill 8B and Qwen3-8B by up to 9.0 and 6.2 absolute Pass@1 scores on LiveCodeBench. Also, we analyze LLMs' in-context reward hacking behaviors and augment Cobalt training with perturbed trajectories to mitigate this issue. Overall, our empirical results demonstrate Cobalt as a promising solution for one-step recoverable tasks like multi-turn code generation.

## Setup

To start, please create a conda environment and install the necessary packages as follows:
```
conda create -n cobalt python=3.10
conda activate cobalt

cd verl
USE_MEGATRON=0 bash scripts/install_vllm_sglang_mcore.sh
pip install opentelemetry-exporter-prometheus==0.47b0
pip install ray==2.49.2
pip install --no-deps -e .
pip install "click<8.3.0"
pip uninstall pynvml
pip install nvidia-ml-py

# Optional: We access Claude via AWS Bedrock
pip install boto3 botocore
```

## Datasets and Preprocessing

### TACO
We provide our processed TACO dataset on Huggingface: [osunlp/TACO-Cobalt](https://huggingface.co/datasets/osunlp/TACO-Cobalt), which contains the cleaned training and validation data, and [osunlp/TACO-Cobalt-PTB](https://huggingface.co/datasets/osunlp/TACO-Cobalt-PTB), which contains the perturbed validation data. We also release the synthesized offline trajectories for both LLMs [here](https://buckeyemailosu-my.sharepoint.com/:f:/g/personal/chen_8336_buckeyemail_osu_edu/IgAKScSv5V8xQLvCeGnv56TiAYeKYRxOr4t6Bzunm6eG6dg?e=FluDv2).

To process the original [TACO-verified](https://huggingface.co/datasets/likaixin/TACO-verified) data from scratch, you may run the following commands to start the code execution server and run the preprocessing script:
```
ray start --head
serve run remote_reward_server:app --name evaluator --route-prefix /evaluator &
python preproc/preproc_taco.py
```

To perturb public test cases for a dataset, you can run the following script:
```
python preproc/run_perturbation.py \
    --input_file {INPUT_FNAME} \
    --output_file {OUTPUT_FNAME}
```
- `input_file`: the dataset file with public test cases.
- `output_file`: the file to save with perturbed test cases.

To generate your own dataset of code generationt trajectories, after running some LLM with our multi-turn [inference](#inference) and [evaluation](#evaluation) procedures, you may run this script to reproduce our dynamic sampling and max-variance down-sampling:
```
python preproc/process_multi_turn_taco.py \
    --input_file {INPUT_FNAME} \
    --output_file {OUTPUT_FNAME} \
    --model_name {LLM_NAME} \
    [--is_test]
```
- `input_file`: the file with all of the inference and evaluation results of an LLM.
- `output_file`: the file with all final trajectories ready for online bandit learning.
- `model_name`: the name or directory of the LLM checkpoint that contains its tokenizer, which is used to truncate overlong prompts.
- `is_test`: whether the input_file is generated with a validation/test set. If enabled, it will only select one trajectory with the lowest test case improvement per task, and the resulting dataset should be used for validation during training.

### LiveCodeBench
Since the hidden test cases in LiveCodeBench are encoded to prevent data leakage, we do not redistribute any of our processed data. To reproduce our experiments, you may first download `test6.jsonl` from [livecodebench/code_generation_lite](https://huggingface.co/datasets/livecodebench/code_generation_lite) and then run the following script to process the data:
```
python preproc/preproc_lcb_test6.py
```

## Training

We include a copy of [veRL v0.5.0](https://github.com/volcengine/verl/tree/v0.5.0) under `cobalt/verl`, which is used in this work for RL training. There are three custom files in this directory:
```
OSU-NLP-Group/cobalt
├───  verl
│    ├─── verl
│    │    ├─── workers
│    │    │    ├─── reward_manager
│    │    │    │    ├─── prime.py
│    ├─── code_reward_func_think.py
│    ├─── remote_reward_server.py
```
where `remote_reward_server.py` is the script to host code execution servers, `code_reward_func_think.py` implements the reward calculations, and `prime.py` is modified to accommodate asynchronous code execution. 

To reproduce Cobalt, we provide two example scripts under `scripts` for reference: `train_grpo_single_turn.sh` for single-step fine-tuning and `train_grpo_multi_turn.sh` for contextual bandit learning.


## Inference

### Open-Weight LLMs
We use [vLLM](https://github.com/vllm-project/vllm) to host open-weight LLMs locally with two Ray replicas for efficient asynchronous inference. To start, you need to modify `vllm_engine_actor.py` with the correct model name or directory path that you plan to run.

For single-turn inference, you can uses the following commands:
```
ray start --head
serve run remote_reward_server:app --name evaluator --route-prefix /evaluator &
serve run vllm_engine_actor:app --name vllm_actor --route-prefix /vllm_actor &

python -u run_single_turn_vllm.py \
    --input_file {INPUT_FNAME} \
    --output_file {OUTPUT_FNAME} \
    --max_tokens 6144
```
For multi-turn inference, you can similarly run:
```
ray start --head
serve run remote_reward_server:app --name evaluator --route-prefix /evaluator &
serve run vllm_engine_actor:app --name vllm_actor --route-prefix /vllm_actor &

python -u run_multi_turn_vllm.py \
    --input_file {INPUT_FNAME} \
    --output_file {OUTPUT_FNAME} \
    --max_tokens 6144 \
    [--enable_thinking]
```
- `input_file`: the input file with evaluation data.
- `output_file`: the output file to save inference results.
- `max_tokens`: number of maximum response tokens.
- `enable_thinking`: if enabled, we will split model response by "\</think\>" to remove the long chain-of-thoughts in multi-turn inference and always limit the response to the first 6000 characters in case any pathological behavior happens.

We note that our multi-turn inference implementation assumes that you have already ran single-turn inference to get the initial programs. This is because, for the need of fair comparisons in our experiments, we separate the first turn out to cache a fixed set of initial programs, which provide the same starting point in various multi-turn experiments (e.g., original data v.s. perturbed data) with the same model. If this is undesired, you can always modify the multi-turn inference code with a few lines of changes to run the procedure end-to-end.

### Proprietary LLMs
We use [LiteLLM](https://github.com/BerriAI/litellm) as a unified interface to access the APIs. Similar to open-weight models, you can run single-turn inference with:
```
ray start --head
serve run remote_reward_server:app --name evaluator --route-prefix /evaluator &

python run_single_turn_litellm.py \ 
    --model_name {OPENAI_OR_BEDROCK_MODEL} \
    --input_file {INPUT_FNAME} \
    --output_file {OUTPUT_FNAME} \
    --openai_api_key {YOUR_OPENAI_KEY} \
    --aws_access_key_id {YOUR_AWS_KEY_ID} \
    --aws_secret_access_key {YOUR_AWS_KEY} \
    --aws_region_name us-west-2
```
and multi-turn inference with:
```
ray start --head
serve run remote_reward_server:app --name evaluator --route-prefix /evaluator &

python run_multi_turn_litellm.py \ 
    --model_name {OPENAI_OR_BEDROCK_MODEL} \
    --input_file {INPUT_FNAME} \
    --output_file {OUTPUT_FNAME} \
    --openai_api_key {YOUR_OPENAI_KEY} \
    --aws_access_key_id {YOUR_AWS_KEY_ID} \
    --aws_secret_access_key {YOUR_AWS_KEY} \
    --aws_region_name us-west-2
```
- `model_name`: the name of proprietary LLM to run.
- `input_file`: the input file with evaluation data.
- `output_file`: the output file to save inference results.
- `openai_api_key`: your OpenAI access key.
- `aws_access_key_id`: your AWS credentials.
- `aws_secret_access_key`: your AWS credentials.

## Evaluation

### Multi-Turn Trajectory Evaluation

Since the multi-turn inference procedure only executes the programs on the public test cases, to obtain their true performance, we need to go through the trajectories again and get the Pass@1 results on the hidden test cases:
```
ray start --head
serve run remote_reward_server:app --name evaluator --route-prefix /evaluator &

python eval_multi_turn.py \
    --input_file {INPUT_FNAME} \
    --output_file {OUTPUT_FNAME}
```
- `input_file`: the input file with LLM multi-turn inference trajectory data.
- `output_file`: the output file to save the hidden test case results.

### In-Context Reward Hacking Analysis

After running multi-turn inference and evaluation on the perturbed dataset [osunlp/TACO-Cobalt-PTB](https://huggingface.co/datasets/osunlp/TACO-Cobalt-PTB), you can reproduce our analysis with LLM judge in two steps. First, you can extract the turn-level errors using:
```
python extract_turn_level_errors.py \
    --data_file {DATA_FNAME} \
    --input_file {INPUT_FNAME} \
    --output_file {OUTPUT_FNAME}
```
- `data_file`: the original perturbed dataset.
- `input_file`: the input file with LLM multi-turn inference trajectory data and evaluation results.
- `output_file`: the output file to save the extracted errors.

Then, you can run the LLM judge on the output file from the previous step:
```
python run_llm_judge.py \ 
    --model_name gpt-5-2025-08-07 \
    --input_file {INPUT_FNAME} \
    --output_file {OUTPUT_FNAME} \
    --openai_api_key {YOUR_OPENAI_KEY}
```
- `model_name`: the name of proprietary LLM to run.
- `input_file`: the input file with turn-level error data.
- `output_file`: the output file to save LLM error classification results.
- `openai_api_key`: your OpenAI access key.


## Contact

[Ziru Chen](mailto:chen.8336@osu.edu), [Yujia Xie](mailto:Xie.Yujia000@gmail.com), [Huan Sun](mailto:sun.397@osu.edu)

## Citation

If you find our code and data useful, please cite our paper:

```
TBA
```