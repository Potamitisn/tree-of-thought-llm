import os
from openai import OpenAI
import backoff 

completion_tokens = prompt_tokens = 0


def completions_with_backoff(**kwargs):
    client = OpenAI()
    return client.chat.completions.create(**kwargs)

def gpt(prompt, model="gpt-4", temperature=0.7, max_tokens=1000, n=1, stop=None) -> list:
    messages = [{"role": "user", "content": prompt}]
    return chatgpt(messages, model=model, temperature=temperature, max_tokens=max_tokens, n=n, stop=stop)
    
def chatgpt(messages, model="gpt-4", temperature=0.7, max_tokens=1000, n=1, stop=None) -> list:
    global completion_tokens, prompt_tokens
    outputs = []
    while n > 0:
        cnt = min(n, 20)
        n -= cnt
        res = completions_with_backoff(model=model, messages=messages, temperature=temperature, max_tokens=max_tokens, n=cnt, stop=stop)
        outputs.extend([choice.message.content for choice in res.choices])
        # log completion tokens
        completion_tokens += res.usage.completion_tokens
        prompt_tokens += res.usage.prompt_tokens
    return outputs
    
def gpt_usage(backend="gpt-4"):
    global completion_tokens, prompt_tokens
    if backend == "gpt-4":
        cost = completion_tokens / 1000 * 0.06 + prompt_tokens / 1000 * 0.03
    elif backend == "gpt-3.5-turbo-1106":
        cost = completion_tokens / 1000 * 0.002 + prompt_tokens / 1000 * 0.001
    elif backend == "gpt-3.5-turbo-0125":
        cost = completion_tokens / 1000 * 0.0015 + prompt_tokens / 1000 * 0.0005
    return {"completion_tokens": completion_tokens, "prompt_tokens": prompt_tokens, "cost": cost}
