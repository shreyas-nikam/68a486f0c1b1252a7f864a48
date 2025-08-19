import pytest
from definition_db2c4a58704446959da03cc39303715c import ingest_documents

@pytest.mark.parametrize("file_paths, expected_type", [
    ([], (list, list)),
    (['test.pdf'], (list, list)),
    (['test.html'], (list, list)),
    (['test.pdf', 'test.html'], (list, list)),
    (['nonexistent_file.txt'], (list, list)),
])
def test_ingest_documents_returns_tuple_of_lists(file_paths, expected_type, monkeypatch):
    # Mock file reading to avoid actual file system interaction.
    def mock_read_file(filepath):
        return "mock content"
    
    monkeypatch.setattr("your_module.ingest_documents", lambda x: ([""], [""]))
    
    result = ingest_documents(file_paths)
    assert isinstance(result, tuple)
    assert isinstance(result[0], list)
    assert isinstance(result[1], list)
