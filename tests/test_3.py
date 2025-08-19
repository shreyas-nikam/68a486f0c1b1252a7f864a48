import pytest
import numpy as np
import faiss
from definition_04e91051f6d649ebae3ddb547818561c import build_faiss_index

def test_build_faiss_index_valid_input():
    embeddings = np.float32([[1, 2, 3], [4, 5, 6]])
    dimension = 3
    index = build_faiss_index(embeddings, dimension)
    assert isinstance(index, faiss.IndexFlatL2)

def test_build_faiss_index_empty_embeddings():
    embeddings = np.empty((0, 128), dtype=np.float32)
    dimension = 128
    index = build_faiss_index(embeddings, dimension)
    assert isinstance(index, faiss.IndexFlatL2)

def test_build_faiss_index_single_embedding():
    embeddings = np.float32([[1, 2, 3]])
    dimension = 3
    index = build_faiss_index(embeddings, dimension)
    assert isinstance(index, faiss.IndexFlatL2)

def test_build_faiss_index_high_dimension():
    embeddings = np.random.rand(10, 1000).astype(np.float32)
    dimension = 1000
    index = build_faiss_index(embeddings, dimension)
    assert isinstance(index, faiss.IndexFlatL2)

def test_build_faiss_index_non_float32_embeddings():
    embeddings = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float64)
    dimension = 3
    with pytest.raises(TypeError):
        build_faiss_index(embeddings, dimension)
