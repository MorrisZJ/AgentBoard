import openai
import os
import time
from common.registry import registry
import pdb
import tiktoken
import json

@registry.register_llm("gpt")
class OPENAI_GPT:
    def __init__(self,
                 engine="gpt-3.5-turbo-0631",
                 temperature=0,
                 max_tokens=200,
                 use_azure=True,
                 top_p=1,
                 stop=["\n"],
                 retry_delays=60, # in seconds
                 max_retry_iters=5,
                 context_length=4096,
                 system_message=''
                 ):
        
        
        self.engine = engine
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.use_azure = use_azure
        self.top_p = top_p
        self.stop = stop
        self.retry_delays = retry_delays
        self.max_retry_iters = max_retry_iters
        self.context_length = context_length
        self.system_message = system_message
        self.init_api_key()

    # #region agent log
    def _agent_debug_log(self, hypothesis_id, location, message, data=None, run_id="pre-fix"):
        """Lightweight NDJSON logger for debug mode."""
        log_path = "/home/mz81/.cursor/debug-7ff043.log"
        payload = {
            "sessionId": "7ff043",
            "id": f"log_{int(time.time()*1000)}",
            "timestamp": int(time.time()*1000),
            "location": location,
            "message": message,
            "data": data or {},
            "runId": run_id,
            "hypothesisId": hypothesis_id,
        }
        try:
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(payload) + "\n")
        except Exception:
            # Never let logging break the main flow
            pass
    # #endregion agent log

        
    def init_api_key(self):
        if self.use_azure:
            openai.api_type = os.environ['OPENAI_API_TYPE']
            openai.api_version = os.environ['OPENAI_API_VERSION']
        else:
            if 'OPENAI_API_KEY' not in os.environ:
                raise Exception("OPENAI_API_KEY environment variable not set.")
            else:
                openai.api_key = os.environ['OPENAI_API_KEY']
            openai.api_base = os.environ['OPENAI_API_BASE'] if 'OPENAI_API_BASE' in os.environ else openai.api_base

    def llm_inference(self, messages):
        # Newer models (e.g. gpt-4o-mini, gpt-5-*) use max_completion_tokens.
        kwargs = dict(
            model=self.engine,
            messages=messages,
        )

        # Some newer models (including gpt-5-* / 2025-* family) no longer
        # support the `stop` or non-default `temperature` parameters.
        is_2025_family = self.engine.startswith("gpt-5-") or "2025" in self.engine

        if not is_2025_family:
            kwargs["stop"] = self.stop
            kwargs["temperature"] = self.temperature

        if (
            self.engine.startswith("gpt-5-")
            or "2025" in self.engine
            or "gpt-4o-mini" in self.engine
        ):
            kwargs["max_completion_tokens"] = self.max_tokens
        else:
            kwargs["max_tokens"] = self.max_tokens

        response = openai.ChatCompletion.create(**kwargs)
        return response['choices'][0]['message']['content']

    def generate(self, system_message, prompt):
        # print(f"{self.num_tokens_from_messages(prompt, self.engine)} prompt tokens counted by num_tokens_from_messages().")

        # self.llm_inference(prompt)
        prompt=[
            {"role": "system", "content": system_message},
            {"role": "user", "content": prompt}
        ]
        for attempt in range(self.max_retry_iters):
            try:
                return True, self.llm_inference(prompt) # return success, completion
            except Exception as e:
                # Print the underlying error so it is visible in logs
                print(e)
                print(f"Error on attempt {attempt + 1}")
                # #region agent log
                self._agent_debug_log(
                    hypothesis_id="H1",
                    location="openai_gpt.py:generate",
                    message="LLM call failed",
                    data={
                        "attempt": attempt + 1,
                        "engine": self.engine,
                        "error": str(e),
                    },
                    run_id="pre-fix",
                )
                # #endregion agent log
                if attempt < self.max_retry_iters - 1:  # If not the last attempt
                    time.sleep(self.retry_delays)  # Wait before retrying

                else:
                    print("Failed to get completion after multiple attempts.")
                    # Keep returning failure flag instead of crashing the whole run

        return False, None

    def num_tokens_from_messages(self, messages, model="gpt-3.5-turbo-0613"):
        """Return the number of tokens used by a list of messages."""
        model = self.engine
        try:
            encoding = tiktoken.encoding_for_model(model)
        except KeyError:
            print("Warning: model not found. Using cl100k_base encoding.")
            encoding = tiktoken.get_encoding("cl100k_base")
        
        tokens_per_message = 0
        tokens_per_name = 0
        if model in {
            "gpt-3.5-turbo-0613",
            "gpt-3.5-turbo-16k-0613",
            "gpt-4-0314",
            "gpt-4-32k-0314",
            "gpt-4-0613",
            "gpt-4-32k-0613",
            }:
            tokens_per_message = 3
            tokens_per_name = 1
        
        num_tokens = 0
        for message in messages:
            num_tokens += tokens_per_message
            for key, value in message.items():
                num_tokens += len(encoding.encode(value))
                if key == "name":
                    num_tokens += tokens_per_name
        num_tokens += 3  # every reply is primed with <|start|>assistant<|message|>
        return num_tokens

    @classmethod
    def from_config(cls, config):
        
        engine = config.get("engine", "gpt-35-turbo")
        temperature = config.get("temperature", 0)
        max_tokens = config.get("max_tokens", 100)
        system_message = config.get("system_message", "You are a helpful assistant.")
        use_azure = config.get("use_azure", True)
        top_p = config.get("top_p", 1)
        stop = config.get("stop", ["\n"])
        retry_delays = config.get("retry_delays", 10)
        context_length = config.get("context_length", 4096)
        return cls(engine=engine,
                   temperature=temperature,
                   max_tokens=max_tokens,
                   use_azure=use_azure,
                   top_p=top_p,
                   retry_delays=retry_delays,
                   system_message=system_message,
                   context_length=context_length,
                   stop=stop)
