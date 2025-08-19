import pytest
from definition_6dec7055356c4440bee282fc0ef04d26 import bm25_retrieval

@pytest.mark.parametrize("query, documents, top_k, expected", [
    ("example query", ["document 1", "example query document", "another document"], 2, [(1, pytest.approx(0.693147, 0.0001)), (0, pytest.approx(0.0, 0.0001))]),
    ("query", [], 1, []),
    ("query", ["document"], 0, []),
    ("complex query with many terms", ["complex query", "another document", "query with some terms"], 2, [(0, pytest.approx(0.693147, 0.0001)), (2, pytest.approx(0.287682, 0.0001))]),
    ("query", ["query"], 1, [(0, pytest.approx(1.098612, 0.0001))]),
])
def test_bm25_retrieval(query, documents, top_k, expected):
    result = bm25_retrieval(query, documents, top_k)
    assert len(result) == len(expected)
    for i in range(len(result)):
        assert result[i][0] == expected[i][0]
        assert result[i][1] == expected[i][1]
