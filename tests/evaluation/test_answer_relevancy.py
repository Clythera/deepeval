import pytest
from app.llm import ask_llm
from tests.evaluation.metrics import AnswerRelevancyMetric
#from tests.evaluation.dataset import get_evaluation_cases
from deepeval.test_case import LLMTestCase
from deepeval import evaluate
from deepeval.models import LocalModel

@pytest.mark.asyncio
async def test_answer_relevancy():
    #input text for the test case
    input_text = "What if these shoes don't fit?"

    # Get the answer from the LLM
    response = await ask_llm(input_text)

    # Initialize your local model (Judge Model) for evaluation
    #if you are using LMStudio to run the local model, you can use the following configuration
    local_model = LocalModel(
        model="openai/gpt-oss-20b",  # Replace with your local model name
        base_url="http://localhost:1234/v1",
        temperature=0,
    )

    metric = AnswerRelevancyMetric(
        threshold=0.7,
        model=local_model,
        include_reason=True
    )
    test_case = LLMTestCase(
        input=input_text,
        # output from your LLM app
        actual_output = response
    )
    # To run metric as a standalone
    metric.measure(test_case)
    print("\n======Output")
    print("\nLLM Response =========", response)
    print(metric.score, metric.reason)

