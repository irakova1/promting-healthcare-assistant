import os
import re
from typing import Optional

from deepeval.models.base_model import DeepEvalBaseLLM
from langchain_community.llms.llamafile import Llamafile
from langchain_core.language_models import BaseChatModel
from langchain_google_vertexai import ChatVertexAI


class LangChatModel(DeepEvalBaseLLM):
    def __init__(self, model: BaseChatModel, model_name: str):

        self.model = model
        self.model_name = model_name

    def load_model(self):
        return self.model

    def generate(self, prompt: str) -> str:
        chat_model = self.load_model()
        return chat_model.invoke(prompt).content

    async def a_generate(self, prompt: str) -> str:
        chat_model = self.load_model()
        res = await chat_model.ainvoke(prompt)
        return res.content

    def get_model_name(self):
        return self.model_name


class Mistral7B(DeepEvalBaseLLM):
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer

    def load_model(self):
        return self.model

    def generate(self, prompt: str) -> str:
        model = self.load_model()

        # the device to load the model onto
        # device = "cuda"

        model_inputs = self.tokenizer([prompt], return_tensors="pt").to(device)
        model.to(device)

        generated_ids = model.generate(**model_inputs, max_new_tokens=100, do_sample=True)
        return self.tokenizer.batch_decode(generated_ids)[0]

    async def a_generate(self, prompt: str) -> str:
        return self.generate(prompt)

    def get_model_name(self):
        return "Mistral 7B"


def get_eval_model(model_name: str, model_url: Optional[str] = None) -> DeepEvalBaseLLM | None:
    region = os.environ['GOOGLE_LOCATION']
    

    match model_name:
        case 'vertexai':
            llm = ChatVertexAI(max_output_tokens=200, location=region)
            return LangChatModel(llm, llm.model_name)

        case path if path is not None and (name := re.findall('([\w.-]+)\.llamafile$', path)):
            llm = Llamafile(base_url=model_url)
            return LangChatModel(llm, name[-1])

        case _:
            print("Unknown eval model, falback to OpenAI")
            # OpenAi used if None returned
            return None
