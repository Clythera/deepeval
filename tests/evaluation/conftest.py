# This is the configuration file for evaluation tests
import pytest
from app.llm import ask_llm

@pytest.fixture
def real_llm_model():
    return ask_llm