import os
from datetime import datetime
import backoff

# APIs
from openai import AzureOpenAI
from together import Together
from groq import Groq


# HF
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

completion_tokens = prompt_tokens = 0

class ClientAPI:
    def __init__(self, backend, max_tokens, temperature=0.7):
        self.backend = backend
        self.temperature = temperature
        self.completion_tokens = self.prompt_tokens = 0
        self.max_tokens = max_tokens

    def gpt(self, prompt, n=1, stop=None) -> list:
        messages = [{"role": "user", "content": prompt}]
        return self.chatgpt(messages, n=n, stop=stop)
        
    def chatgpt(self, messages, n=1, stop=None) -> list:
        outputs = []
        while n > 0:
            cnt = min(n, 20)
            n -= cnt
            # Rearranged it a bit to account for updated OpenAI API
            #res = completions_with_backoff(model=model, messages=messages, temperature=temperature, max_tokens=max_tokens, n=cnt, stop=stop)
            #outputs.extend([choice.message.content for choice in res.choices])
            # log completion tokens
            #completion_tokens += res.usage.completion_tokens
            #prompt_tokens += res.usage.prompt_tokens
            outputs.extend(self.completions_with_backoff(messages=messages, n=cnt, stop=stop))
        return outputs
    
    def completions_with_backoff(self, messages, n, stop):
        pass

    def gpt_usage(backend):
        pass

class ClientTogetherAI(ClientAPI):
    def completions_with_backoff(self, messages, n, stop):
        # Get the api key for TogetherAI
        access_token = "TOGETHER_API_KEY_DLAB"
        api_key = os.getenv(access_token)
        assert api_key, f"Access token '{access_token}' not found in environment variables!"

        # Setup the client
        client = Together(api_key=api_key)

        
        # Make the request
        response = client.chat.completions.create(
            model=self.backend,
            messages=messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            n=n,
            stop=stop)
        
        # Get the outputs 
        outputs = [(choice.message.content, response.usage.completion_tokens/n, response.usage.prompt_tokens) for choice in response.choices]

        # Update the tokens usage
        self.completion_tokens += sum(output[1] for output in outputs)
        self.prompt_tokens += outputs[0][2]
        

        
        # Get the message contents
        message_contents = [output[0] for output in outputs]
        
        # print the message contents
        m = f"+++PROMPT+++\n{messages[0]['content']}\n\n" + f"+++RESPONSES+++ :\n" + '----------------------\n'.join([r+'\n' for r in message_contents])
        print("#"*100 + "\n" + m + "\n" + "%"*100)

        return message_contents
    
    def gpt_usage(self, backend):
        assert backend == self.backend

        # LLama 3.1
        if backend == "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo":
            cost = (self.completion_tokens + self.prompt_tokens) / 1000000 * 0.18
        elif backend == "meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo":
            cost = (self.completion_tokens + self.prompt_tokens) / 1000000 * 0.88
        elif backend == "meta-llama/Meta-Llama-3.1-405B-Instruct-Turbo":
            cost = (self.completion_tokens + self.prompt_tokens) / 1000000 * 5

        # LLama 3.2
        elif backend == "meta-llama/Llama-3.2-11B-Vision-Instruct-Turbo":
            cost = (self.completion_tokens + self.prompt_tokens) / 1000000 * 0.18
        elif backend == "meta-llama/Llama-Vision-Free":
            cost = (self.completion_tokens + self.prompt_tokens) / 1000000 * 0.18
        elif backend == "meta-llama/Llama-3.2-90B-Vision-Instruct-Turbo":
            cost = (self.completion_tokens + self.prompt_tokens) / 1000000 * 1.2
        

       # Mixtral 
        elif backend == "mistralai/Mixtral-8x22B-Instruct-v0.1":
            cost = (self.completion_tokens + self.prompt_tokens) / 1000000 * 1.2
        elif backend == "mistralai/Mixtral-8x7B-Instruct-v0.1":
            cost = (self.completion_tokens + self.prompt_tokens) / 1000000 * 0.6
        else:
            raise ValueError(f"Unknown backend: {backend}")

        return {"completion_tokens": self.completion_tokens, "prompt_tokens": self.prompt_tokens, "cost": cost}

class ClientGroq(ClientAPI):
    def completions_with_backoff(self, messages, n, stop):
        global groq_completion_tokens, groq_prompt_tokens
        
        # Get the api key for Groq
        access_token = "GROQ_API_KEY"
        api_key = os.getenv(access_token)
        assert api_key, f"Access token '{access_token}' not found in environment variables!"

        # Setup the client
        client = Groq(api_key=api_key)

        # Make the request
        response = client.chat.completions.create(
            model=self.backend,
            messages=messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            n=n,
            stop=stop)
        
        # Get the outputs 
        outputs = [(choice.message.content, response.usage.completion_tokens/n, response.usage.prompt_tokens) for choice in response.choices]

        # Update the tokens usage
        self.completion_tokens += sum(output[1] for output in outputs)
        self.prompt_tokens += outputs[0][2]
        
        # Get the message contents
        message_contents = [output[0] for output in outputs]
        return message_contents
    
    def gpt_usage(self, backend):
        # Groq is free for now but we consider the on-demand pricing for tokens-as-a-service (https://groq.com/pricing/)
        assert backend == self.backend
        if backend == "llama-3.1-70b-versatile":
            cost = self.completion_tokens / 1000000 * 0.79 + self.prompt_tokens / 1000000 * 0.59
        elif backend == "llama-3.1-8b-instant":
            cost = self.completion_tokens / 1000000 * 0.08 + self.prompt_tokens / 1000000 * 0.05
        elif backend == "llama3-70b-8192":
            cost = self.completion_tokens / 1000000 * 0.79 + self.prompt_tokens / 1000000 * 0.59
        elif backend == "llama3-8b-8192":
            cost = self.completion_tokens / 1000000 * 0.08 + self.prompt_tokens / 1000000 * 0.05
        elif backend == "mixtral-8x7b-32768":
            cost = self.completion_tokens / 1000000 * 0.24 + self.prompt_tokens / 1000000 * 0.24
        else:
            raise ValueError(f"Unknown backend: {backend}")

        return {"completion_tokens": self.completion_tokens, "prompt_tokens": self.prompt_tokens, "cost": cost}
    


    
class ClientOpenAI(ClientAPI):
    def completions_with_backoff(self, messages, n, stop):
        global completion_tokens, prompt_tokens
        model2keys = {
            "gpt-4-0613": {"access_token": "AZURE_OPENAI_KEY2LOC1", "endpoint": "key-2"},
            "gpt-35-turbo-0125" : {"access_token":"AZURE_OPENAI_KEY2LOC2", "endpoint": "key-2-loc2"},
            "gpt-4-0125-preview": {"access_token":"AZURE_OPENAI_KEY2LOC3", "endpoint": "key-2-loc3"},
            "gpt-4o-2024-05-13-global": {"access_token": "AZURE_OPENAI_KEY1LOC1", "endpoint": "key-1-18k"},
        }
        
        access_token = model2keys.get(self.backend, {}).get("access_token", False)
        endpoint = model2keys.get(self.backend, {}).get("endpoint", False)
        assert bool(access_token and endpoint), f"Model {self.backend} not supported!"

        api_key = os.getenv(access_token)
        assert api_key, f"Access token '{access_token}' not found in environment variables!"

        client = AzureOpenAI(
            azure_endpoint="https://"+endpoint+".openai.azure.com/",
            api_key=api_key,
            api_version="2024-02-15-preview")
        
        print("#"*50 + f"\n{messages[0]['content']}\n" + "%"*50)
        response = client.chat.completions.create(
            model=self.backend,
            messages=messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            n=n,
            stop=stop)

        
        outputs = [(choice.message.content, response.usage.completion_tokens/n, response.usage.prompt_tokens) for choice in response.choices]

        # AzureOpenAI has a strong filtering mechanism, so we need to retry if some of the choices are filtered and therefore None
        for retry_count in range(5):
            if any([output[0] is None for output in outputs]):
                print(f"Some choices are None, retrying (retry : {retry_count+1})")
                outputs = [output for output in outputs if output[0] is not None]
                n_none = n - len(outputs)
                response = client.chat.completions.create(
                    model=self.backend,
                    messages=messages,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                    n=n_none,
                    stop=stop)
                outputs.extend((choice.message.content, response.usage.completion_tokens/n, response.usage.prompt_tokens) for choice in response.choices)
        completion_tokens += sum(output[1] for output in outputs)
        prompt_tokens += outputs[0][2]
        message_contents = [output[0] for output in outputs]
        return message_contents
        
    def gpt_usage(self, backend="gpt-4"):
        assert backend == self.backend
        global completion_tokens, prompt_tokens
        if backend == "gpt-4" or backend =="gpt-4-0613":
            cost = completion_tokens / 1000 * 0.06 + prompt_tokens / 1000 * 0.03
        elif backend == "gpt-3.5-turbo" or backend == "gpt-3.5-turbo-0125" or backend == "gpt-35-turbo-0125":
            cost = completion_tokens / 1000 * 0.0015 + prompt_tokens / 1000 * 0.0005
        elif backend == "gpt-4-0125-preview":
            cost = completion_tokens / 1000 * 0.03 + prompt_tokens / 1000 * 0.01
        elif backend == "gpt-4o-2024-05-13-global":
            cost = completion_tokens / 1000 * 0.015 + prompt_tokens / 1000 * 0.005
        else:
            raise ValueError(f"Unknown backend: {backend}")

        return {"completion_tokens": completion_tokens, "prompt_tokens": prompt_tokens, "cost": cost}


class ClientHF:

    def __init__(self, model_dir, temperature=0.7):
        self.completion_tokens = self.prompt_tokens = 0
        self.start_time = datetime.now()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device : {self.device}")

        self.model_name = model_dir.split("/")[-1]
        self.temperature = temperature
        hf_token=None
        load_in_8bit=True
        device_map='auto'

        self.tokenizer = AutoTokenizer.from_pretrained(model_dir, use_auth_token=hf_token)
        self.model = AutoModelForCausalLM.from_pretrained(model_dir, use_auth_token=hf_token,
                                                            device_map=device_map,
                                                            load_in_8bit=load_in_8bit)
        print(f'Model {model_dir.split("/")[-1]} loaded successfully')

    def gpt(self, prompt, max_tokens=100, n=1, stop=None) -> list:
        messages = [{"role": "user", "content": prompt}]
        return self.chatgpt(messages, max_tokens=max_tokens, n=n, stop=stop)
        
    def chatgpt(self, messages, max_tokens=100, n=1, stop=None) -> list:
        outputs = []
        while n > 0:
            cnt = min(n, 20)
            n -= cnt
            res = self.completions_with_backoff(messages=messages, max_tokens=max_tokens, n=cnt, stop=stop)
            outputs.extend([choice["message"]["content"] for choice in res["choices"]])
            
            # log completion tokens
            self.completion_tokens += res["usage"]["completion_tokens"]
            self.prompt_tokens += res["usage"]["prompt_tokens"]
        return outputs
    
    #@backoff.on_exception(backoff.expo, openai.error.OpenAIError)
    def completions_with_backoff(self, messages, max_tokens, n, stop):
        # Deviated from **kwargs configuration but made sure all the required parameters are there
        # <model> parameter not really needed, just keeping there for consistency
        inputs = self.tokenizer(messages[0]["content"], return_tensors="pt").to(self.device)
        print(f"Generating a completion...")
        outputs = self.model.generate(**inputs,
                              max_new_tokens=max_tokens,
                              num_return_sequences=n,
                              temperature=self.temperature,
                              stop_strings=stop,
                              tokenizer=self.tokenizer)
        
        # Removing the input prompt from the output
        reducted_outputs = outputs[:, inputs["input_ids"].shape[1]:]
        responses = [self.tokenizer.decode(output, skip_special_tokens=True) for output in reducted_outputs]

        completion_tokens = sum(self.tokenizer(response, return_tensors="pt")["input_ids"].shape[1] for response in responses)
        prompt_tokens = inputs["input_ids"].shape[1]

        # Format reponse in OpenAI's style for consistency
        formatted_response = {"choices":[{"index":i, "message":{"content":response, "role":"assistant"}}] for i, response in enumerate(responses) }
        formatted_response["usage"] = {"prompt_tokens":prompt_tokens, "completion_tokens": completion_tokens}

        return formatted_response
    
    def gpt_usage(self, backend=None):
        elapsed_time = str(datetime.now() - self.start_time)
        return {"completion_tokens": self.completion_tokens, "prompt_tokens": self.prompt_tokens, "elapsed_time": elapsed_time}