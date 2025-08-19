import pytest
from definition_fe899eb8fa5143b7af0b1a5a9a1fa153 import evaluate_groundedness

@pytest.mark.parametrize("answer, context, expected", [
    ("The answer is in the context.", "The answer is in the context.", 1.0),
    ("The answer is not in the context.", "Some other context.", 0.0),
    ("Partial answer.", "Partial answer. But also other info", 0.5),
    ("", "Some context.", 0.0),
    ("Answer.", "", 0.0),
])
def test_evaluate_groundedness(answer, context, expected):
    assert evaluate_groundedness(answer, context) == expected
