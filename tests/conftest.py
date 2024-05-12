import pytest


@pytest.fixture(scope="session")
def model_name(pytestconfig):
    return pytestconfig.getoption("model_name")


@pytest.fixture(scope="session")
def model_url(pytestconfig):
    return pytestconfig.getoption("model_url")


def pytest_addoption(parser):
    parser.addoption("--model_name", action="store", default=None)
    parser.addoption("--model_url", action="store", default='http://127.0.0.1:8081')


def pytest_generate_tests(metafunc):
    # This is called for every test. Only get/set command line arguments
    # if the argument is specified in the list of test "fixturenames".
    option_value = metafunc.config.option.model_name
    if 'model_name' in metafunc.fixturenames and option_value is not None:
        metafunc.parametrize("model_name", [option_value])

    option_value_model_url = metafunc.config.option.model_url
    if 'model_url' in metafunc.fixturenames and option_value is not None:
        metafunc.parametrize("model_url", [option_value_model_url])
