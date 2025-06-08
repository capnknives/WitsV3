"""Test configuration and common fixtures."""
import pytest
import asyncio
import tempfile
from pathlib import Path
from typing import Generator

@pytest.fixture(scope="session")
def event_loop() -> Generator[asyncio.AbstractEventLoop, None, None]:
    """Create an event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()

@pytest.fixture
def temp_dir() -> Generator[Path, None, None]:
    """Create a temporary directory for test files."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        yield Path(tmp_dir)

@pytest.fixture
def sample_data() -> dict:
    """Provide sample data for testing."""
    return {
        "numbers": [1, 2, 3, 4, 5],
        "text": "Hello, World!",
        "nested": {
            "key": "value",
            "array": [{"id": 1}, {"id": 2}]
        }
    }

@pytest.fixture
def mock_httpx_response():
    """Mock httpx response for testing."""
    class MockResponse:
        def __init__(self, json_data, status_code=200):
            self._json = json_data
            self.status_code = status_code

        def json(self):
            return self._json

    return MockResponse

@pytest.fixture
def mock_async_client():
    """Mock async HTTP client for testing."""
    class MockAsyncClient:
        def __init__(self, response_data=None, status_code=200):
            self.response_data = response_data
            self.status_code = status_code

        async def get(self, *args, **kwargs):
            return self.mock_response()

        async def post(self, *args, **kwargs):
            return self.mock_response()

        def mock_response(self):
            class MockResponse:
                def __init__(self, data, status):
                    self._data = data
                    self.status_code = status

                async def json(self):
                    return self._data

            return MockResponse(self.response_data, self.status_code)

    return MockAsyncClient 