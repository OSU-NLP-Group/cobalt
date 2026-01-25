from vllm import LLM, SamplingParams

import ray
from ray import serve

NUM_GPUS = 4
NUM_ENGINES = 2

@serve.deployment(
    num_replicas=NUM_ENGINES,
    max_replicas_per_node=NUM_ENGINES,
    ray_actor_options={"num_cpus": 1, "num_gpus": NUM_GPUS // NUM_ENGINES},
    max_ongoing_requests=1,
)
class VllmEngineActor:
    def __init__(self):
        llm_kwargs = {
            "model": "Qwen/Qwen3-8B",
            "tensor_parallel_size": NUM_GPUS // NUM_ENGINES,
            "enforce_eager": True,
        }
        
        self.llm = LLM(**llm_kwargs)
        self.tokenizer = self.llm.get_tokenizer()
    
    async def __call__(self, messages, temperature=0.6, top_p=0.95, max_new_tokens=8192, n=16, is_batch=False):
        if is_batch:
            prompts = []
            for msg in messages:
                prompt = self.tokenizer.apply_chat_template(
                    msg,
                    add_generation_prompt=True,
                    tokenize=False,
                )
                prompts.append(prompt)
        else:
            prompts = self.tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=False,
            )

        sampling_params = SamplingParams(
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_new_tokens,
            n=n,
        )

        output = self.llm.generate(prompts, sampling_params, use_tqdm=False)

        if is_batch:
            responses = [ou.outputs[0].text for ou in output]
        else:
            responses = [o.text for o in output[0].outputs]

        return {
            "prompts": prompts,
            "responses": responses,
        }

app = VllmEngineActor.bind()