import pytest
from definition_4a14b601430747c199baecec296f53c7 import chunk_text

@pytest.mark.parametrize("text, chunk_size, chunk_overlap, expected", [
    ("This is a test sentence.", 5, 1, ['This ', 'is a ', 'a tes', 'test ', 'senten', 'entenc', 'tence.']),
    ("This is a test.", 10, 0, ['This is a ', 'test.']),
    ("This is a test.", 5, 2, ['This ', 'is is', 's is ', 'is a ', 'a tes', 'test.']),
    ("", 5, 1, []),
    ("Short", 10, 0, ['Short'])
])
def test_chunk_text(text, chunk_size, chunk_overlap, expected):
    assert chunk_text(text, chunk_size, chunk_overlap) == expected
