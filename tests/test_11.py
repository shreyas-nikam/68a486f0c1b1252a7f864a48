import pytest
from definition_47ba4dfd962448cb90ae23c941bb125c import evaluate_answer_quality

@pytest.mark.parametrize("question, answer, model_name, expected", [
    ("What is the capital of France?", "Paris is the capital of France.", "rouge", 1.0),  # Ideal answer
    ("What is the capital of France?", "I don't know.", "rouge", 0.0),  # No knowledge
    ("What is the capital of France?", "France is a country.", "rouge", 0.2),  # Partially relevant
    ("", "", "rouge", 0.0),  # Empty inputs
    ("What is the capital of France?", "Paris is the capital of France.", "unknown_model", ValueError),  # Wrong model name
])

def test_evaluate_answer_quality(question, answer, model_name, expected):
    try:
        if model_name == "rouge":
            #Mock the rouge score
            if answer == "Paris is the capital of France.":
                assert evaluate_answer_quality(question, answer, model_name) == 1.0
            elif answer == "I don't know.":
                assert evaluate_answer_quality(question, answer, model_name) == 0.0
            elif answer == "France is a country.":
                assert evaluate_answer_quality(question, answer, model_name) == 0.2
            elif question == "" and answer == "":
                assert evaluate_answer_quality(question, answer, model_name) == 0.0
        else:
           evaluate_answer_quality(question, answer, model_name)
    except Exception as e:
        assert isinstance(e, type(expected))
