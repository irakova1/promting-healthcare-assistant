from deepeval.models import DeepEvalBaseLLM

from tests.models import get_eval_model


class SingletonMeta(type):
    """
    The Singleton class will use the metaclass because it is best suited for this purpose.
    """

    _instances = {}

    def __call__(cls, *args, **kwargs):
        """
        Possible changes to the value of the `__init__` argument do not affect
        the returned instance.
        """
        if cls not in cls._instances:
            instance = super().__call__(*args, **kwargs)
            cls._instances[cls] = instance
        return cls._instances[cls]


class SetupTest(metaclass=SingletonMeta):
    def __init__(self, model_name: str, model_url: str):
        self.eval_model = get_eval_model(model_name, model_url=model_url)

    def get_eval_model(self) -> DeepEvalBaseLLM | None:
        """
        Return single instance of the chat model from the first creation of this class.
        """
        return self.eval_model
