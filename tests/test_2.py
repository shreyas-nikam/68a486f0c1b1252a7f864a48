import pytest
from definition_669b88b951404676aed737808741edd0 import embed_text
import numpy as np
from unittest.mock import patch

@pytest.fixture
def mock_embedding():
    return np.array([0.1, 0.2, 0.3])

@pytest.mark.parametrize("text_chunks, model_name, mock_embedding_return, expected_shape", [
    (["This is a test."], "test_model", [np.array([0.1, 0.2, 0.3])], (1, 3)),
    (["This is chunk 1", "This is chunk 2"], "test_model", [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]], (2, 3)),
    ([], "test_model", [], (0,)),
    ([""], "test_model", [[]], (1,0)),
])

def test_embed_text(text_chunks, model_name, mock_embedding_return, expected_shape):
    with patch('sentence_transformers.SentenceTransformer.encode') as mock_encode:
        mock_encode.side_effect = mock_embedding_return
        embeddings = embed_text(text_chunks, model_name)

        if len(text_chunks) > 0 and len(mock_embedding_return[0])>0:
            assert isinstance(embeddings, np.ndarray)
            assert embeddings.shape == expected_shape
        elif len(text_chunks) == 0:
            assert isinstance(embeddings, np.ndarray)
            assert embeddings.shape == expected_shape
        elif len(text_chunks) > 0 and len(mock_embedding_return[0])==0:
            assert isinstance(embeddings, np.ndarray)
            assert embeddings.shape == expected_shape