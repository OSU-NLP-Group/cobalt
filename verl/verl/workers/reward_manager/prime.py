# Copyright 2024 PRIME team and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import asyncio
from copy import deepcopy
# from concurrent.futures import ThreadPoolExecutor # ProcessPoolExecutor
from functools import partial
from typing import Any, Callable, Optional

import psutil
import torch
from transformers import PreTrainedTokenizer

from verl import DataProto
from verl.utils.reward_score import default_compute_score
from verl.workers.reward_manager import register
from verl.workers.reward_manager.abstract import AbstractRewardManager

import random


async def single_compute_score(evaluation_func, completion, reference, task, task_extra_info, sem, timeout=300.0):
    async with sem:
        try:
            coro = evaluation_func(task, completion, reference, task_extra_info)
            result = await asyncio.wait_for(coro, timeout=timeout)
            return result
        except asyncio.TimeoutError:
            print(f"[Timeout] Task timeout: {completion[:80]}")
            return None  # Default value for timed-out rows
        except Exception as e:
            print(f"[Error] Task failed: {e}, completion: {completion[:80]}")
            return None  # Default value for failed rows


async def parallel_compute_score_async(
    evaluation_func, completions, references, tasks, extra_info=None, num_processes=64
):
    if extra_info is None:
        extra_info = [None] * len(tasks)
    scores = []

    sem = asyncio.Semaphore(num_processes)
    tasks_async = []
    for c, r, t, ei in zip(completions, references, tasks, extra_info, strict=True):
        tasks_async.append(
            asyncio.create_task(
                single_compute_score(evaluation_func, c, r, t, ei, sem, timeout=300.0)
            )
        )
    results = await asyncio.gather(*tasks_async, return_exceptions=False)

    # Process results
    for result, completion, reference, task in zip(results, completions, references, tasks, strict=True):
        if isinstance(result, Exception) or result is None:
            # Handle failed or timed-out tasks
            scores.append(-1.0)
        elif isinstance(result, int | float | bool):
            scores.append(float(result))
        else:
            scores.append(float(result[0]))
    return scores


def run_reward_scoring(evaluation_func, completions, references, tasks, extra_info=None, num_processes=64):
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        return loop.run_until_complete(
            parallel_compute_score_async(evaluation_func, completions, references, tasks, extra_info, num_processes)
        )
    finally:
        loop.close()


@register("prime")
class PrimeRewardManager(AbstractRewardManager):
    """
    The Reward Manager used in https://github.com/PRIME-RL/PRIME
    """

    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        num_examine: int,
        compute_score: Optional[Callable] = None,
        reward_fn_key: str = "data_source",
    ) -> None:
        self.tokenizer = tokenizer
        self.num_examine = 1 # num_examine  # the number of batches of decoded responses to print to the console
        self.compute_score = compute_score or default_compute_score
        self.reward_fn_key = reward_fn_key
        self.is_val = (num_examine == 1)

    def verify(self, data):
        """
        verify the batch and save as ``acc`` tensor
        """
        # batched scoring
        prompt_ids = data.batch["prompts"]

        response_ids = data.batch["responses"]
        sequences_str = self.tokenizer.batch_decode(response_ids, skip_special_tokens=True)
        ground_truth = [data_item.non_tensor_batch["reward_model"]["ground_truth"] for data_item in data]
        data_sources = data.non_tensor_batch[self.reward_fn_key]
        extra_info = data.non_tensor_batch.get("extra_info", None)

        _extra_info = deepcopy(extra_info)  # list of dict
        if _extra_info is None:
            _extra_info = [{} for _ in range(len(sequences_str))]
        for ei in _extra_info:
            ei["is_val"] = self.is_val

        assert len(sequences_str) == len(ground_truth) == len(data_sources)
        try:
            scores = run_reward_scoring(
                self.compute_score,
                completions=sequences_str,
                references=ground_truth,
                tasks=data_sources,
                extra_info=_extra_info,
                num_processes=64,
            )
        except asyncio.TimeoutError:
            print("[Timeout] Global reward scoring timed out. Setting all as 0.")
            scores = [0.0 for _ in range(len(sequences_str))]
        except Exception as e:
            print(f"[Error] Unexpected error during scoring. Setting all as 0. {e}")
            scores = [0.0 for _ in range(len(sequences_str))]
        data.batch["acc"] = torch.tensor(scores, dtype=torch.float32, device=prompt_ids.device)
        return scores

    def __call__(self, data: DataProto, return_dict: bool = False) -> torch.Tensor | dict[str, Any]:
        """We will expand this function gradually based on the available datasets"""

        # If there is rm score, we directly return rm score. Otherwise, we compute via rm_score_fn
        if "rm_scores" in data.batch.keys():
            if return_dict:
                reward_extra_keys = data.meta_info.get("reward_extra_keys", [])
                reward_extra_info = {key: data.non_tensor_batch[key] for key in reward_extra_keys}
                return {"reward_tensor": data.batch["rm_scores"], "reward_extra_info": reward_extra_info}
            else:
                return data.batch["rm_scores"]

        reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)

        already_print_data_sources = {}

        # batched scoring
        prompt_ids = data.batch["prompts"]
        prompt_length = prompt_ids.shape[-1]

        response_ids = data.batch["responses"]
        valid_response_length = data.batch["attention_mask"][:, prompt_length:].sum(dim=-1)
        sequences_str = self.tokenizer.batch_decode(response_ids, skip_special_tokens=True)
        data_sources = data.non_tensor_batch["data_source"]

        scores = self.verify(data)

        for i in range(len(data)):
            data_source = data_sources[i]
            reward_tensor[i, valid_response_length[i].item() - 1] = scores[i]

            if data_source not in already_print_data_sources:
                already_print_data_sources[data_source] = 0

            if already_print_data_sources[data_source] < self.num_examine:
                already_print_data_sources[data_source] += 1
                # print(sequences_str)
                
                examine_idx = random.randint(0, len(sequences_str) - 1)
                print([scores[examine_idx], sequences_str[examine_idx]])

        if return_dict:
            return {"reward_tensor": reward_tensor}
        else:
            return reward_tensor