import pytest
from definition_8cc414a8623a46e8a9bf8df24e6c5f35 import create_rag_prompt

@pytest.mark.parametrize("query, retrieved_chunks, expected", [
    ("What is the capital of France?", ["Paris is the capital.", "France is a country."], "What is the capital of France?\nContext:\nParis is the capital.\nFrance is a country."),
    ("Tell me about the weather.", [], "Tell me about the weather.\nContext:\n"),
    ("Summarize this document", ["This is a long document.", "It contains important information."], "Summarize this document\nContext:\nThis is a long document.\nIt contains important information."),
    ("Question with special chars?", ["Answer with \"quotes\"", "And some & ampersands"], "Question with special chars?\nContext:\nAnswer with \"quotes\"\nAnd some & ampersands"),
    ("", ["Context"], "\nContext:\nContext"),
])

def test_create_rag_prompt(query, retrieved_chunks, expected):
    assert create_rag_prompt(query, retrieved_chunks) == expected
