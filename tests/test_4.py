import pytest
import numpy as np
from unittest.mock import MagicMock
from definition_b609ed7b4b2c40c0ac7d30d315207f27 import query_index

@pytest.fixture
def mock_faiss_index():
    index = MagicMock()
    return index

def test_query_index_valid_input(mock_faiss_index):
    mock_faiss_index.search.return_value = (np.array([[1.0, 0.5]]), np.array([[1, 2]]))
    query_embedding = np.array([0.1, 0.2]).astype('float32')
    top_k = 2
    results = query_index(mock_faiss_index, query_embedding, top_k)
    assert len(results) == 2
    assert results[0] == [1, 1.0]
    assert results[1] == [2, 0.5]
    mock_faiss_index.search.assert_called_once()

def test_query_index_empty_index(mock_faiss_index):
    mock_faiss_index.search.return_value = (np.array([[]]), np.array([[]]))
    query_embedding = np.array([0.1, 0.2]).astype('float32')
    top_k = 2
    results = query_index(mock_faiss_index, query_embedding, top_k)
    assert results == []

def test_query_index_top_k_zero(mock_faiss_index):
    query_embedding = np.array([0.1, 0.2]).astype('float32')
    top_k = 0
    results = query_index(mock_faiss_index, query_embedding, top_k)
    assert results == []

def test_query_index_invalid_query_embedding_type(mock_faiss_index):
    with pytest.raises(AttributeError):
        query_embedding = [0.1, 0.2]
        top_k = 2
        query_index(mock_faiss_index, query_embedding, top_k)

def test_query_index_top_k_larger_than_index(mock_faiss_index):
    mock_faiss_index.search.return_value = (np.array([[1.0, 0.5]]), np.array([[1, 2]]))
    query_embedding = np.array([0.1, 0.2]).astype('float32')
    top_k = 5
    results = query_index(mock_faiss_index, query_embedding, top_k)
    assert len(results) == 2
    assert results[0] == [1, 1.0]
    assert results[1] == [2, 0.5]
    mock_faiss_index.search.assert_called_once()
