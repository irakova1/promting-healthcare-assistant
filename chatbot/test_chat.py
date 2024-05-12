import argparse
import os
import sys
import time

from dotenv import load_dotenv

sys.path.insert(0, os.path.abspath(os.path.dirname(os.path.dirname(__file__))))

from chat import Chat, ModelType
from langchain.globals import set_verbose, set_debug
from promt import DEFAULT_INSTRUCT_MEDICAL_TEMPLATE


def chat_test(
    model_type: ModelType,
    db_url: str,
    model_url: str,
    temperature: float,
    use_buffer: bool,
):
    load_dotenv(override=True)
    # set_verbose(True)
    # set_debug(True)

    chat = Chat.get_chat(
        model_type,
        db_url,
        model_url,
        temperature=temperature,
        prompt_config=dict(
            use_buffer=use_buffer,
            prompt=DEFAULT_INSTRUCT_MEDICAL_TEMPLATE,
        ),
    )
    print("Model name:", chat.model_name)
    print("Temp:", temperature)

    chat.clear_session_memory("aaaa")

    q0 = "How can bloodpresure impact my sight?"
    q1 = "Shorten your initial answer."

    start_time = time.time()
    answer0 = chat.get_answer(user_id="aaaa", question=q0)
    answer1 = chat.get_answer(user_id="aaaa", question=q1)
    end_time = time.time()

    print("Question:", q0)
    print("Answer type:", type(answer0))
    print('Answer: """', answer0, '"""')

    print("\n\n", "=" * 32, "\n")

    print("Question:", q1)
    print("Answer type:", type(answer1))
    print('Answer: """', answer1, '"""')

    chat.clear_session_memory("aaaa")

    print("\n\n", "=" * 32, "\n")
    print("Time:", end_time - start_time, "s")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model_type",
        type=ModelType,
        choices=ModelType,
        default=ModelType.LLAMAFILE,
        help="Type of model will be used in Chat to evaluate.",
    )
    parser.add_argument(
        "--model_url",
        type=str,
        default="http://127.0.0.1:8080",
        help="URL to llamafile based model eg http://127.0.0.1:8081",
    )
    parser.add_argument(
        "--db_url",
        type=str,
        default="sqlite:///sqlite.db",
        help="URL to messages database.",
    )

    parser.add_argument("--temp", type=float, default=0.0, help="Temperature.")

    args = parser.parse_args()

    chat_test(
        args.model_type,
        args.db_url,
        model_url=args.model_url,
        temperature=args.temp,
        use_buffer=False,  # FIXME: add args, default False
    )

    # python chatbot/test_chat.py --model_type huggingface --model_url epfl-llm/meditron-7b --temp 0.1
    # python chatbot/test_chat.py --model_type huggingface --model_url mistralai/Mistral-7B-v0.1 --temp 0.1

    # python chatbot/test_chat.py --model_type openai-gpt3 --temp 0.1

    # llamafile-0.7.exe -m .\BioMistral-7B.Q4_K_M.llamafile.zip --server --nobrowser --embedding -ngl 999
    # python chatbot/test_chat.py --model_type llamafile --model_url http://127.0.0.1:8080 --temp 0.1
