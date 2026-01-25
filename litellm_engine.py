import litellm
from litellm import completion
litellm.drop_params = True

import os
import time
import random

class LiteLlmEngine():
    def __init__(self, model_name, model_args_dict):
        self.model_name = model_name

        if self.model_name.startswith("bedrock"):
            os.environ["AWS_ACCESS_KEY_ID"] = model_args_dict["aws_access_key_id"]
            os.environ["AWS_SECRET_ACCESS_KEY"] = model_args_dict["aws_secret_access_key"]
            os.environ["AWS_REGION_NAME"] = model_args_dict["aws_region_name"]
        else:
            os.environ["OPENAI_API_KEY"] = model_args_dict["openai_api_key"]


    def respond(self, messages, reasoning_effort=None, temperature=1.0, top_p=1.0, max_tokens=32000):
        time.sleep(random.uniform(0.5, 2.0))

        responses = completion(
            model=self.model_name,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            reasoning_effort=reasoning_effort,
            timeout=900,
            max_retries=3,
        )

        return {
            "response": responses.choices[0].message.content, 
            "usage": {
                "prompt_tokens": responses.usage.prompt_tokens,
                "completion_tokens": responses.usage.completion_tokens,
                "reasoning_tokens": responses.usage.completion_tokens_details.reasoning_tokens if responses.usage.completion_tokens_details is not None else 0,
            }
        }
