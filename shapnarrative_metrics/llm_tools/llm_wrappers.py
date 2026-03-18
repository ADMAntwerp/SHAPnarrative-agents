from abc import ABC, abstractmethod
from openai import OpenAI
from openai._types import NOT_GIVEN
import anthropic
from mistralai import Mistral
from google import genai
from google.genai import types
import time


class LLMWrapper(ABC):
    @abstractmethod
    def generate_response(self, prompt, history):
        """
        Generates a response to the given prompt.

        :param prompt: The input prompt to generate a response for.
        :return: The generated response as a string.
        """
        pass

class ClaudeApi(LLMWrapper):

    def __init__(
        self,
        api_key,
        model,
        system_role="You are a teacher, skilled at explaining complex AI decisions to general audiences.",
        temperature=0.6,
    ):
        
        self.system_role = system_role
        self.model = model 
        self.client = anthropic.Anthropic(api_key=api_key)
        self.messages=[]
        self.temperature=temperature

    def generate_response(self, prompt, history=[]):
        message = self.client.messages.create(
            model=self.model,
            max_tokens=1000,
            temperature=self.temperature,
            system=self.system_role,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": prompt
                        }
                    ]
                }
            ]
        )

        return message.content[0].text
            

class GptApi(LLMWrapper):

    def __init__(
        self,
        api_key,
        model,
        system_role="You are a teacher, skilled at explaining complex AI decisions to general audiences.",
        temperature=0.6
    ):
        self.system_role = system_role
        self.model = model 
        self.client = OpenAI(api_key=api_key)
        self.messages=[]
        self.temperature=temperature

    def generate_response(self, prompt, history=[]):

        messages=history+[{ "role": "system", "content": self.system_role }, {"role": "user", "content": prompt}]
        completion = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=self.temperature
        )

        return completion.choices[0].message.content

class OpenRouterAPI:
    def __init__(
        self,
        api_key,
        model,
        system_role="You are a teacher, skilled at explaining complex AI decisions to general audiences.",
        temperature=0.6,
        max_retries=3,  
        retry_wait=2 
    ):

        self.api_key = api_key
        self.model = model
        self.system_role = system_role
        self.temperature=temperature
        self.client = OpenAI(base_url="https://openrouter.ai/api/v1",api_key=api_key,)
        self.max_retries = max_retries
        self.retry_wait = retry_wait

    def generate_response(self, prompt, history=[]):

        messages=history+[{ "role": "system", "content": self.system_role }, {"role": "user", "content": prompt}]
        
        # print out messaged to see why it failed and gives several more tries
        for attempt in range(1, self.max_retries + 1):
            try:
                completion = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=self.temperature
                )

                if not completion or not getattr(completion, "choices", None):
                    raise ValueError("⚠️ OpenRouter returned no choices.")

                message = completion.choices[0].message
                if not message or not getattr(message, "content", None):
                    raise ValueError("⚠️ OpenRouter returned an empty message content.")

                return message.content

            except Exception as e:
                print(f"❌ OpenRouterAPI call failed (attempt {attempt}/{self.max_retries}): {e}")
                if attempt < self.max_retries:
                    time.sleep(self.retry_wait)
                else:
                    print("🚨 All retries failed. Returning None.")
                    return None


class MistralApi(LLMWrapper):

    def __init__(
        self,
        api_key,
        model,
        system_role="You are a teacher, skilled at explaining complex AI decisions to general audiences.",
        temperature=0.6,
        max_retries=3,  
        retry_wait=2 
    ):
        
        self.system_role = system_role
        self.model = model 
        self.client = Mistral(api_key=api_key)
        self.messages=[]
        self.temperature=temperature
        self.max_retries = max_retries
        self.retry_wait = retry_wait

    def generate_response(self, prompt, history=[]):
         
        messages=history+[{ "role": "system", "content": self.system_role }, {"role": "user", "content": prompt}]

        for attempt in range(1, self.max_retries + 1):
            try:
                completion = self.client.chat.complete(
                    model= self.model,
                    messages = messages,
                    temperature=self.temperature
                )

                if not completion or not getattr(completion, "choices", None):
                    raise ValueError("⚠️ Mistral returned no choices.")

                message = completion.choices[0].message
                if not message or not getattr(message, "content", None):
                    raise ValueError("⚠️ Mistral returned an empty message content.")

                return message.content

            except Exception as e:
                print(f"❌ MistralAPI call failed (attempt {attempt}/{self.max_retries}): {e}")
                if attempt < self.max_retries:
                    time.sleep(self.retry_wait)
                else:
                    print("🚨 All retries failed. Returning None.")
                    return None


class GoogleApi(LLMWrapper):
    def __init__(
        self,
        api_key,
        model,
        system_role="You are a teacher, skilled at explaining complex AI decisions to general audiences.",
        temperature=0.6
    ):
        self.system_role = system_role
        self.model = model
        self.temperature = temperature
        self.client = genai.Client(api_key=api_key) 
        self.messages = []

    def generate_response(self, prompt, history=[]):
        contents = [
            types.Content(
                role="user",
                parts=[types.Part.from_text(text=prompt)]
            )
        ]

        generate_content_config = types.GenerateContentConfig(
       temperature=self.temperature
    )
        response = self.client.models.generate_content(
            model=self.model,
            contents=contents,
            config=generate_content_config,
        )
        
        return response.text

class DeepSeekApi(LLMWrapper):

    def __init__(
        self,
        api_key,
        model,
        system_role="You are a teacher, skilled at explaining complex AI decisions to general audiences.",
        temperature=0.6,
        base_url="https://api.deepseek.com/v1",
    ):
        self.system_role = system_role
        self.model = model
        self.messages = []
        self.temperature = temperature
        self.base_url = base_url
        self.client = OpenAI(api_key=api_key, base_url=self.base_url)

    def generate_response(self, prompt):

        messages = [
            {"role": "system", "content": self.system_role},
            {"role": "user", "content": prompt},
        ]
        completion = self.client.chat.completions.create(
            model=self.model, messages=messages, temperature=self.temperature
        )

        return completion.choices[0].message.content