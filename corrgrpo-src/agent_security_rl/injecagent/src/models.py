import os
from src.utils import get_response_text
import time


class BaseModel:
    def __init__(self):
        self.model = None

    def prepare_input(self, sys_prompt,  user_prompt_filled):
        raise NotImplementedError("This method should be overridden by subclasses.")

    def call_model(self, model_input):
        raise NotImplementedError("This method should be overridden by subclasses.")
    
class ClaudeModel(BaseModel):     
    def __init__(self, params):
        super().__init__()  
        from anthropic import Anthropic, HUMAN_PROMPT, AI_PROMPT
        self.anthropic = Anthropic(
            api_key= os.environ.get("ANTHROPIC_API_KEY"),
        )
        self.params = params
        self.human_prompt = HUMAN_PROMPT
        self.ai_prompt = AI_PROMPT

    def prepare_input(self, sys_prompt, user_prompt_filled):
        model_input = f"{self.human_prompt} {sys_prompt} {user_prompt_filled}{self.ai_prompt}"
        return model_input

    def call_model(self, model_input):
        completion = self.anthropic.completions.create(
            model=self.params['model_name'],
            max_tokens_to_sample=4096,
            prompt=model_input,
            temperature=0
        )
        return completion.completion
    
class GPTModel(BaseModel):
    def __init__(self, params):
        super().__init__()  
        from openai import OpenAI
        client_kwargs = {
            "api_key": os.environ.get("OPENAI_API_KEY"),
            "max_retries": int(os.environ.get("OPENAI_MAX_RETRIES", "8")),
            "timeout": float(os.environ.get("OPENAI_TIMEOUT", "180")),
        }
        if os.environ.get("OPENAI_ORGANIZATION"):
            client_kwargs["organization"] = os.environ["OPENAI_ORGANIZATION"]
        if os.environ.get("OPENAI_BASE_URL"):
            client_kwargs["base_url"] = os.environ["OPENAI_BASE_URL"]
        self.client = OpenAI(**client_kwargs)
        self.params = params

    def prepare_input(self, sys_prompt, user_prompt_filled):
        model_input = [
            {"role": "system", "content":sys_prompt},
            {"role": "user", "content": user_prompt_filled}
        ]
        return model_input

    def call_model(self, model_input):
        request_delay = float(os.environ.get("INJECAGENT_API_REQUEST_DELAY", "0"))
        if request_delay > 0:
            time.sleep(request_delay)
        completion = self.client.chat.completions.create(
            model=self.params['model_name'],
            messages=model_input,
            temperature=0,
            max_tokens=int(os.environ.get("INJECAGENT_MAX_TOKENS", "1024")),
        )
        message = completion.choices[0].message
        return message.content or getattr(message, "reasoning_content", None) or ""
    
class TogetherAIModel(BaseModel): 
    def __init__(self, params):
        super().__init__()  
        import together
        from src.prompts.prompt_template import PROMPT_TEMPLATE
        
        self.together = together
        self.params = params
        self.model = self.params['model_name']
        self.prompt = PROMPT_TEMPLATE[self.model]

    def prepare_input(self, sys_prompt, user_prompt_filled):
        model_input = self.prompt.format(sys_prompt = sys_prompt, user_prompt = user_prompt_filled)
        return model_input
    
    def call_model(self, model_input, retries=3, delay=2, max_tokens =512):
        attempt = 0
        while attempt < retries:
            try:
                completion = self.together.Complete.create(
                    model=self.model,
                    prompt=model_input,
                    max_tokens=max_tokens,
                    temperature=0
                )
                return completion['choices'][0]['text']
            except Exception as e:
                attempt += 1
                max_tokens = max_tokens // 2
                print(f"Attempt {attempt}: An error occurred - {e}")
                if attempt < retries:
                    time.sleep(delay)  # Wait for a few seconds before retrying
                else:
                    return ""
    
class LlamaModel(BaseModel):     
    def __init__(self, params):
        super().__init__()  
        import transformers
        import torch
        tokenizer = transformers.AutoTokenizer.from_pretrained(params['model_name'])
        self.pipeline = transformers.pipeline(
            "text-generation",
            model=params['model_name'],
            tokenizer=tokenizer,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            max_length=4096,
            do_sample=False, # temperature=0
            eos_token_id=tokenizer.eos_token_id,
        )

    def prepare_input(self, sys_prompt, user_prompt_filled):
        model_input = f"[INST] <<SYS>>\n{sys_prompt}\n<</SYS>>\n\n{user_prompt_filled} [/INST]"
        return model_input

    def call_model(self, model_input):
        output = self.pipeline(model_input)
        return get_response_text(output, "[/INST]")
    
MODELS = {
    "Claude": ClaudeModel,
    "GPT": GPTModel,
    "Llama": LlamaModel,
    "TogetherAI": TogetherAIModel
}   
