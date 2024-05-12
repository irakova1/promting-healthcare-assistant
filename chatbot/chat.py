import os
import torch
from abc import ABC
from enum import Enum
from typing import Optional, List, Tuple

from langchain.chains.conversation.base import ConversationChain
from langchain.memory import ConversationSummaryBufferMemory
from langchain_anthropic import ChatAnthropic
from langchain_community.chat_message_histories.sql import SQLChatMessageHistory
from langchain_community.llms.llamafile import Llamafile
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import convert_to_messages
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import BasePromptTemplate
from langchain_core.runnables import ConfigurableFieldSpec
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_google_vertexai import ChatVertexAI
from langchain_google_vertexai import HarmBlockThreshold, HarmCategory
from langchain_mistralai.chat_models import ChatMistralAI
from langchain_openai import ChatOpenAI
from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline
from transformers import pipeline as hf_pipeline, AutoTokenizer
from transformers import StoppingCriteriaList

from chatbot.promt import MSG_END, DEFAULT_MEDICAL_TEMPLATE, DEFAULT_MEDICAL_CONVERSATION_PROMPT, DEFAULT_INSTRUCT_MEDICAL_TEMPLATE
from chatbot.stop_string_criteria import StopStringCriteria


class ModelType(Enum):
    VERTEXAI = "vertexai"
    VERTEXAI_BISON = "vertexai-bison"
    VERTEXAI_GEMINI_PRO = "vertexai-gemini-pro"
    ANTHROPIC_OPUS = "anthropic-claude-3-opus"
    ANTHROPIC_SONNET = "anthropic-claude-3-sonnet"
    ANTHROPIC_HAIKU = "anthropic-claude-3-haiku"
    MISTRAL_LARGE = "mistral-large"
    MISTRAL_MEDIUM = "mistral-medium"
    MISTRAL_SMALL = "mistral-small"
    MISTRAL_MIXTRAL = "mistral-mixtral"
    MISTRAL_7B = "mistral-7b"
    OPENAI_GPT3 = "openai-gpt3"
    OPENAI_GPT4 = "openai-gpt4"
    OPENAI_GPT4_TURBO = "openai-gpt4-turbo"
    LLAMAFILE = "llamafile"
    HUGGINGFACE = "huggingface"


class ChatI(ABC):
    chain_with_history: RunnableWithMessageHistory
    model_name: str
    prompt: BasePromptTemplate
    prompt_config: dict
    temperature: float

    def get_user_history(self, user_id: str) -> BaseChatMessageHistory:
        pass

    def get_answer(self, user_id, question: str) -> str:
        pass

    def clear_session_memory(self, user_id: str) -> None:
        pass

    def add_messages(self, user_id: str, messages: List[Tuple[str, str]]) -> None:
        pass

    @staticmethod
    def get_chat(
        model_type: ModelType, chat_db: str, model_url: Optional[str] = None
    ) -> "ChatI":
        pass


class Chat(ChatI):

    def __init__(
        self,
        model: BaseChatModel,
        database_url: str,
        prompt_config: dict,
        temperature: float,
        model_name: str = "",
        is_instruct: bool = False,
    ):
        self.model = model
        self.DB_URL = database_url

        self.temperature = temperature

        self.prompt_config = prompt_config

        self.is_instruct = is_instruct

        if self.is_instruct:
            self.use_buffer = False
            print("Cant use buffer for instruct")
        else:
            self.use_buffer = prompt_config["use_buffer"]

        if self.use_buffer:
            self.buffer_token_limit = prompt_config["buffer_token_limit"]
            self.buffer_return_messages = prompt_config["buffer_return_messages"]

        self.human_prefix = prompt_config["human_prefix"]
        self.ai_prefix = prompt_config["ai_prefix"]

        self.model_name = model_name

        # after init variables
        self.chain_with_history = self.prepare_chain(prompt_config["prompt"])

    def prepare_chain(self, prompt: Optional[BasePromptTemplate]):
        if self.use_buffer:
            memory_buffer = ConversationSummaryBufferMemory( # FIXME: this buffer is common for all user_ids
                llm=self.model,
                max_token_limit=self.buffer_token_limit,
                return_messages=self.buffer_return_messages,
                human_prefix=self.human_prefix,
                ai_prefix=self.ai_prefix,
                # ai_prefix="<|im_start|>assistant\n",
                # human_prefix="<|im_start|>user\n",
            )
            params = {"prompt": DEFAULT_MEDICAL_CONVERSATION_PROMPT}
            if prompt:
                params.update({"prompt": prompt})

            runnable = ConversationChain(llm=self.model, memory=memory_buffer, **params)
            # save prompt
            prompt = runnable.prompt
        else:
            if not prompt:
                if not self.is_instruct:
                    prompt = DEFAULT_MEDICAL_TEMPLATE
                else:
                    prompt = DEFAULT_INSTRUCT_MEDICAL_TEMPLATE

            output_parser = StrOutputParser()

            runnable = prompt | self.model | output_parser

        # save prompt
        self.prompt = prompt

        with_message_history = RunnableWithMessageHistory(
            runnable,
            self.get_user_history,
            input_messages_key="input",
            output_messages_key="response" if self.use_buffer else None,
            history_messages_key="history",
            history_factory_config=[
                ConfigurableFieldSpec(
                    id="user_id",
                    annotation=str,
                    name="User ID",
                    description="Unique identifier for the user.",
                    default="",
                    is_shared=True,
                ),
            ],
            stop=[MSG_END]
        )

        return with_message_history

    def get_user_history(self, user_id: str) -> SQLChatMessageHistory:
        # TODO: add sampling by  number of messages or tokens
        return SQLChatMessageHistory(session_id=user_id, connection_string=self.DB_URL)

    def clear_session_memory(self, user_id: str) -> None:
        return SQLChatMessageHistory(
            session_id=user_id, connection_string=self.DB_URL
        ).clear()

    def add_messages(self, user_id: str, messages: List[Tuple[str, str]]) -> None:
        if not self.is_instruct:
            messages = [(m[0], m[1] + MSG_END) for m in messages]

        return SQLChatMessageHistory(
            session_id=user_id, connection_string=self.DB_URL
        ).add_messages(convert_to_messages(messages))

    def get_answer(self, user_id, question: str) -> str:
        if not self.is_instruct:
            question = question + MSG_END

        answer = self.chain_with_history.invoke(
            {"input": question},
            config={"configurable": {"user_id": user_id}},
        )

        if self.use_buffer:
            answer = answer["response"]

        return answer.replace(MSG_END, "").strip()

    @staticmethod
    def get_chat(
        model_type: ModelType,
        chat_db: str,
        model_url: Optional[str] = None,
        model_name: Optional[str] = None,
        temperature: float = 0.0,
        is_instruct: bool = False,
        **kwargs,
    ) -> ChatI:
        prompt_config = {
            "use_buffer": True,  # bool
            "buffer_token_limit": 1024,  # int
            "buffer_return_messages": False,  # bool
            "prompt": None,  # Optional[BasePromptTemplate]
            "human_prefix": "Human",  # str
            "ai_prefix": "AI",  # str
        }

        if "prompt_config" in kwargs:
            prompt_config.update(kwargs["prompt_config"])
            del kwargs["prompt_config"]

        match model_type:
            case ModelType.VERTEXAI:
                # default models is chat-bison
                llm = ChatVertexAI(
                    location=os.environ["GOOGLE_LOCATION"],
                    temperature=temperature,
                    **kwargs,
                )
                model_name = "chat-bison"

            case ModelType.VERTEXAI_BISON:
                model_name = "text-bison"
                llm = ChatVertexAI(
                    model_name=model_name,
                    location=os.environ["GOOGLE_LOCATION"],
                    temperature=temperature,
                    **kwargs,
                )

            case ModelType.VERTEXAI_GEMINI_PRO:
                model_name = "gemini-pro"

                safety_settings = {
                    HarmCategory.HARM_CATEGORY_UNSPECIFIED: HarmBlockThreshold.BLOCK_ONLY_HIGH,
                    HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_ONLY_HIGH,
                    HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_ONLY_HIGH,
                    HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_ONLY_HIGH,
                    HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_ONLY_HIGH,
                }
                llm = ChatVertexAI(
                    model_name=model_name,
                    convert_system_message_to_human=True,
                    location=os.environ["GOOGLE_LOCATION"],
                    temperature=temperature,
                    safety_settings=safety_settings,
                    response_validation=False,
                )

            case ModelType.ANTHROPIC_OPUS:
                model_name = os.environ["ANTHROPIC_LATEST_OPUS"]
                llm = ChatAnthropic(
                    temperature=temperature,
                    anthropic_api_key=os.environ["ANTHROPIC_API_KEY"],
                    model_name=model_name,
                )

            case ModelType.ANTHROPIC_SONNET:
                model_name = os.environ["ANTHROPIC_LATEST_SONNET"]
                llm = ChatAnthropic(
                    temperature=temperature,
                    anthropic_api_key=os.environ["ANTHROPIC_API_KEY"],
                    model_name=model_name,
                )

            case ModelType.ANTHROPIC_HAIKU:
                model_name = os.environ["ANTHROPIC_LATEST_HAIKU"]
                llm = ChatAnthropic(
                    temperature=temperature,
                    anthropic_api_key=os.environ["ANTHROPIC_API_KEY"],
                    model_name=model_name,
                )

            case ModelType.MISTRAL_LARGE:
                model_name = os.environ["MISTRAL_LARGE_LATEST"]
                llm = ChatMistralAI(
                    temperature=temperature,
                    mistral_api_key=os.environ["MISTRAL_API_KEY"],
                    model=model_name,
                )

            case ModelType.MISTRAL_MEDIUM:
                model_name = os.environ["MISTRAL_MEDIUM_LATEST"]
                llm = ChatMistralAI(
                    temperature=temperature,
                    mistral_api_key=os.environ["MISTRAL_API_KEY"],
                    model=model_name,
                )

            case ModelType.MISTRAL_SMALL:
                model_name = os.environ["MISTRAL_SMALL_LATEST"]
                llm = ChatMistralAI(
                    temperature=temperature,
                    mistral_api_key=os.environ["MISTRAL_API_KEY"],
                    model=model_name,
                )

            case ModelType.MISTRAL_MIXTRAL:
                model_name = os.environ["MISTRAL_MIXTRAL_LATEST"]
                llm = ChatMistralAI(
                    temperature=temperature,
                    mistral_api_key=os.environ["MISTRAL_API_KEY"],
                    model=model_name,
                )

            case ModelType.MISTRAL_7B:
                model_name = os.environ["MISTRAL_7B_LATEST"]
                llm = ChatMistralAI(
                    temperature=temperature,
                    mistral_api_key=os.environ["MISTRAL_API_KEY"],
                    model=model_name,
                )

            case ModelType.OPENAI_GPT3:
                model_name = os.environ["OPENAI_GPT3_LATEST"]
                llm = ChatOpenAI(
                    model_name=model_name,
                    temperature=temperature,
                    openai_api_key=os.environ["OPENAI_API_KEY"],
                    openai_organization=os.environ["OPENAI_ORGANIZATION_ID"],
                )

            case ModelType.OPENAI_GPT4:
                model_name = os.environ["OPENAI_GPT4_LATEST"]
                llm = ChatOpenAI(
                    model_name=model_name,
                    temperature=temperature,
                    openai_api_key=os.environ["OPENAI_API_KEY"],
                    openai_organization=os.environ["OPENAI_ORGANIZATION_ID"],
                )

            case ModelType.OPENAI_GPT4_TURBO:
                model_name = os.environ["OPENAI_GPT4_TURBO_LATEST"]
                llm = ChatOpenAI(
                    model_name=model_name,
                    temperature=temperature,
                    openai_api_key=os.environ["OPENAI_API_KEY"],
                    openai_organization=os.environ["OPENAI_ORGANIZATION_ID"],
                )

            case ModelType.LLAMAFILE:
                llm = Llamafile(base_url=model_url, temperature=temperature, **kwargs)

            case ModelType.HUGGINGFACE:
                model_kwargs = {
                    "top_p": 0.95,
                    "do_sample": True,
                    "temperature": temperature,
                    "repetition_penalty": 1.05,
                }
                if temperature == 0:
                    model_kwargs["do_sample"] = False
                    del model_kwargs["temperature"]
                    del model_kwargs["top_p"]

                pipeline_kwargs = {
                    "max_new_tokens": 384,#256,
                    "return_full_text": False,
                }

                if "model_kwargs" in kwargs:
                    model_kwargs.update(kwargs["model_kwargs"])

                if "pipeline_kwargs" in kwargs:
                    pipeline_kwargs.update(kwargs["pipeline_kwargs"])

                tokenizer = AutoTokenizer.from_pretrained(model_url, **model_kwargs)

                stopping_criteria = StoppingCriteriaList(
                    [StopStringCriteria(tokenizer, [MSG_END, "\n\n", "</s>"])] # TODO: check all stopwords is needed
                )

                pipe = hf_pipeline(
                    task="text-generation",
                    model=model_url,  # model repo name from HuggingFace
                    tokenizer=tokenizer,
                    device_map="auto",
                    device="cuda:0",
                    max_new_tokens=pipeline_kwargs["max_new_tokens"],
                    torch_dtype=torch.float16,
                    model_kwargs=model_kwargs,
                    return_full_text=False,
                    stopping_criteria=stopping_criteria,
                )
                llm = HuggingFacePipeline(
                    pipeline=pipe,
                    model_id=model_url,
                    model_kwargs=model_kwargs,
                    pipeline_kwargs=pipeline_kwargs,
                    batch_size=4,
                )

                print(f"{model_kwargs=}")
                print(f"{pipeline_kwargs=}")

                model_name = model_url

            case _:
                raise NotImplementedError

        if not model_name:
            model_name = ""

        return Chat(
            llm,
            chat_db,
            prompt_config=prompt_config,
            model_name=model_name,
            temperature=temperature,
            is_instruct=is_instruct,
        )
