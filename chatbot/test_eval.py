import argparse
import os.path
from types import NoneType
from typing import List, Tuple, Union, Callable

import pandas as pd
from dotenv import load_dotenv

import sys
sys.path.insert(0, os.path.abspath(os.path.dirname(os.path.dirname(__file__))))


# load project variables
load_dotenv()
from chat import ModelType, ChatI, Chat
from eval import evaluate_model, compare_rouge
from promt import DEFAULT_INSTRUCT_MEDICAL_TEMPLATE

from langchain.globals import set_debug, set_verbose

set_verbose(True)
set_debug(True)


def eval_test(
    chat_list: List[Tuple[Union[ChatI, Callable[[], ChatI]], str]],
    data_path: str,
    timeout_s: int = 0,
):
    df = pd.read_csv(
        data_path,
    )
    df_name = os.path.basename(data_path)

    assert (
        len(set(df.columns).intersection(["question", "answer"])) >= 2
    ), "data should have columns ['question', 'answer']"

    context = df.get("context", [])
    if type(context) != NoneType and len(context):
        df_name += "_with_context"

    paths = []
    for chat, name in chat_list:
        path = evaluate_model(
            model_name=name,
            chat=chat,
            input_texts=df.question,
            target_outputs=df.answer,
            input_contexts=context,
            timeout=timeout_s,
            dataset_name=df_name,
        )
        paths.append(path)

    res_path = compare_rouge(paths)
    print(res_path)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model_type",
        type=ModelType,
        choices=ModelType,
        # default=ModelType.VERTEXAI,
        default=ModelType.LLAMAFILE,
        help="Type of model will be used in Chat to evaluate.",
    )
    parser.add_argument(
        "--model_url",
        type=str,
        default="http://127.0.0.1:8080",
        help="URL to llamafile based model eg http://127.0.0.1:8080",
    )
    parser.add_argument(
        "--db_url",
        type=str,
        default="sqlite:///sqlite.db",
        help="URL to messages database",
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default="./data/mini_test.csv",
        help="Path to the data in csv for evaluation.",
    )

    args = parser.parse_args()

    chats = []
    t1 = 0.8
    t2 = 0.0

    use_buffer = False

    is_instruct = True

    prompt_config = {"use_buffer": False}

    chats.append(
        # instantionate Chat only durring eval to save VRAM
        (
            lambda: Chat.get_chat(
                args.model_type,
                args.db_url,
                model_url=args.model_url,
                temperature=t1,
                prompt_config=prompt_config,
                is_instruct=is_instruct,
            ),
            args.model_type.value + f"_{t1}_" + str(args.model_url),
        )
    )
    chats.append(
        (
            lambda: Chat.get_chat(
                args.model_type,
                args.db_url,
                model_url=args.model_url,
                temperature=t2,
                prompt_config=prompt_config,
                is_instruct=is_instruct,
            ),
            args.model_type.value + f"_{t2}_" + str(args.model_url),
        )
    )

    eval_test(chats, args.data_path)
