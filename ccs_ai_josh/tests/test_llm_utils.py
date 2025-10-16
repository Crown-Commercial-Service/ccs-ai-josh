from src.llm_utils import check_index_naming

# these classes allow us to test the functions without actually connecting to an index
class MockField:
    def __init__(self, name):
        self.name = name

class MockIndex:
    def __init__(self, fields):
        self.fields = fields

class MockIndexClient:
    def __init__(self, fields):
        self._index = MockIndex(fields)

    def get_index(self, index_name):
        return self._index

def test_check_index_naming():
    # Both fields present
    client = MockIndexClient([MockField('content_vector'), MockField('content')])
    assert check_index_naming(client, 'my_index') is True

    # One field missing
    client = MockIndexClient([MockField('content_vector')])
    assert check_index_naming(client, 'my_index') is False

    # No fields
    client = MockIndexClient([])
    assert check_index_naming(client, 'my_index') is False