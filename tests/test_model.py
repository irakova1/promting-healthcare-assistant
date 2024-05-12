import os

from typing import Optional
import deepeval
import pytest
from deepeval import assert_test
from deepeval.metrics import AnswerRelevancyMetric, HallucinationMetric, ToxicityMetric, BiasMetric
from deepeval.test_case import LLMTestCase
from dotenv import load_dotenv

from chatbot.chat import Chat, ModelType
from tests.setup import SetupTest
from tests.utils import get_dataset, get_deepeval_dataset, prompt_to_readable

# deepEval necessary environment variables
load_dotenv(override=True)
assert os.environ['DEEPEVAL_RESULTS_FOLDER']
# assert os.environ['OPENAI_API_KEY']

EVAL_MODEL: Optional[str] = os.environ.get('EVAL_MODEL')
EVAL_MODEL_URL = None
DATASET_PATH = os.environ['DEEPEVAL_TEST_DATASET']

# load model
chat = Chat.get_chat(ModelType(os.environ['CHAT_MODEL']), os.environ['CHAT_DB'], os.environ['CHAT_MODEL_URL'],
                     prompt_config=dict(use_buffer=False), model_name=os.environ.get('CHAT_MODEL_NAME', None),
                     temperature=float(os.environ.get('TEMPERATURE', 0.0)), is_instruct=bool(os.environ.get('IS_INSTRUCT', False)))
print("Model", os.environ['CHAT_MODEL'], "url:", os.environ['CHAT_MODEL_URL'])
# set up test dataset by making prediction by the project's chat
test_case_dataset = get_deepeval_dataset(get_dataset(DATASET_PATH), os.path.basename(DATASET_PATH), chat,
                                         timeout_s=int(os.environ['API_TIME_OUT']))


# Loop through test cases using Pytest
@pytest.mark.parametrize(
    "test_case",
    test_case_dataset,
)
def test_over_dataset(test_case: LLMTestCase):
    # calling singleton to take over the same eval model
    if EVAL_MODEL is not None and EVAL_MODEL.startswith("gpt-"):
        eval_model = EVAL_MODEL
    else:
        eval_model = SetupTest(EVAL_MODEL, EVAL_MODEL_URL).get_eval_model()

    answer_relevancy_metric = AnswerRelevancyMetric(threshold=0.5, model=eval_model)
    hallucination_metric = HallucinationMetric(threshold=0.5, model=eval_model)
    toxicity_metric = ToxicityMetric(threshold=0.5, model=eval_model)
    bias_metric = BiasMetric(threshold=0.5, model=eval_model)

    assert_test(test_case, [answer_relevancy_metric, hallucination_metric, toxicity_metric, bias_metric])


@deepeval.log_hyperparameters(model=os.environ['CHAT_MODEL'], prompt_template=prompt_to_readable(chat.prompt))
def hyperparameters():
    hyperparams = {
        'eval_model': EVAL_MODEL,
        'temperature': chat.temperature,
        'CHAT_DB': os.environ.get('CHAT_DB', None),
        'API_TIME_OUT': os.environ.get('API_TIME_OUT', None),
        'model_name': chat.model_name,
        'prompt_config': str(chat.prompt_config),
        'actual_prompt': str(chat.prompt),
        'is_instruct': chat.is_instruct,
    }

    return hyperparams
