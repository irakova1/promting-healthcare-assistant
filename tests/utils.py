import time

import pandas as pd
from deepeval.dataset import EvaluationDataset
from deepeval.test_case import LLMTestCase
from langchain_core.prompts import BasePromptTemplate
from tqdm import tqdm

from chatbot.chat import ChatI


def get_dataset(path: str) -> pd.DataFrame:
    """
    Loads and preprocesses dataset to be used at testing.

    :param path: path to the csv file
    :return: DataFrame with columns ['question', 'answer']
    """
    df = pd.read_csv(path, index_col=0)
    necessary_cols = ['question', 'answer', 'context']
    assert len(set(df.columns).intersection(necessary_cols)) >= len(necessary_cols), \
        f"data should have columns {necessary_cols}"

    df.loc[:, necessary_cols] = df.loc[:, necessary_cols].fillna('',)

    return df


def get_deepeval_dataset(df: pd.DataFrame, dataset_alias: str, chat: ChatI, timeout_s: int = 0) -> EvaluationDataset:
    cases = []

    for i in tqdm(df.index):
        # in case of quota on request/second
        time.sleep(timeout_s)

        item = df.loc[i]
        question = item['question']
        answer = item['answer']
        context = item['context'].split('. ')
        session_id = f'test_{i}'

        # clear history
        chat.clear_session_memory(session_id)
        # add context with one message
        if len('. '.join(context)):
            chat.add_messages(session_id, [('human', '. '.join(context)), ('ai', 'Understood. How can I help?')])

        cases.append(LLMTestCase(input=question,
                                 actual_output=chat.get_answer(f'test_{i}', question),
                                 expected_output=answer,
                                 context=context))

    dataset = EvaluationDataset(
        alias=dataset_alias,
        test_cases=cases
    )

    return dataset


def prompt_to_readable(prompt: BasePromptTemplate) -> str:
    history = [('ai', 'Ai message'), ('human', 'Human message')]

    d = {}
    for v in prompt.input_variables:
        d[v] = f'{{{v}}}'
        if prompt.input_types.get(v, None) is not None:
            d[v] = history

    return prompt.format(**d)
