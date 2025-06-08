"""Tests for the JSON tool."""
import pytest
import json
import tempfile
from pathlib import Path
from tools.json_tool import JSONTool

@pytest.fixture
def json_tool():
    """Create a JSONTool instance for testing."""
    return JSONTool()

@pytest.fixture
def sample_json():
    """Create a sample JSON object for testing."""
    return {
        "name": "test",
        "values": [1, 2, 3],
        "nested": {
            "key": "value",
            "array": [{"id": 1}, {"id": 2}]
        }
    }

def test_json_get_value(json_tool, sample_json):
    """Test getting values from JSON."""
    result = json_tool.execute(
        operation="get",
        data=json.dumps(sample_json),
        path="name"
    )
    assert result["success"] is True
    assert result["value"] == "test"

    result = json_tool.execute(
        operation="get",
        data=json.dumps(sample_json),
        path="nested.key"
    )
    assert result["success"] is True
    assert result["value"] == "value"

    result = json_tool.execute(
        operation="get",
        data=json.dumps(sample_json),
        path="values[0]"
    )
    assert result["success"] is True
    assert result["value"] == 1

def test_json_set_value(json_tool, sample_json):
    """Test setting values in JSON."""
    result = json_tool.execute(
        operation="set",
        data=json.dumps(sample_json),
        path="name",
        value="new_name"
    )
    assert result["success"] is True
    assert json.loads(result["data"])["name"] == "new_name"

    result = json_tool.execute(
        operation="set",
        data=json.dumps(sample_json),
        path="nested.new_key",
        value="new_value"
    )
    assert result["success"] is True
    assert json.loads(result["data"])["nested"]["new_key"] == "new_value"

def test_json_merge(json_tool, sample_json):
    """Test merging JSON objects."""
    merge_data = {
        "name": "merged",
        "new_field": "value"
    }
    
    result = json_tool.execute(
        operation="merge",
        data=json.dumps(sample_json),
        merge_data=json.dumps(merge_data)
    )
    assert result["success"] is True
    merged = json.loads(result["data"])
    assert merged["name"] == "merged"
    assert merged["new_field"] == "value"
    assert "values" in merged

def test_json_validate(json_tool):
    """Test JSON validation."""
    valid_json = '{"name": "test"}'
    result = json_tool.execute(
        operation="validate",
        data=valid_json
    )
    assert result["success"] is True
    assert result["is_valid"] is True

    invalid_json = '{"name": test}'
    result = json_tool.execute(
        operation="validate",
        data=invalid_json
    )
    assert result["success"] is True
    assert result["is_valid"] is False

def test_json_format(json_tool):
    """Test JSON formatting."""
    unformatted = '{"name":"test","values":[1,2,3]}'
    result = json_tool.execute(
        operation="format",
        data=unformatted
    )
    assert result["success"] is True
    assert "\n" in result["formatted"]
    assert "    " in result["formatted"]

def test_json_file_operations(json_tool, sample_json):
    """Test JSON file operations."""
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        f.write(json.dumps(sample_json).encode())
        temp_path = f.name

    try:
        # Test read
        result = json_tool.execute(
            operation="read_file",
            file_path=temp_path
        )
        assert result["success"] is True
        assert json.loads(result["data"]) == sample_json

        # Test write
        new_data = {"new": "data"}
        result = json_tool.execute(
            operation="write_file",
            file_path=temp_path,
            data=json.dumps(new_data)
        )
        assert result["success"] is True

        # Verify written data
        with open(temp_path) as f:
            assert json.load(f) == new_data

    finally:
        Path(temp_path).unlink()

def test_json_schema(json_tool):
    """Test JSON tool schema."""
    schema = json_tool.get_schema()
    
    assert schema["name"] == "json"
    assert "operation" in schema["parameters"]["properties"]
    assert "data" in schema["parameters"]["properties"]
    assert schema["parameters"]["required"] == ["operation", "data"] 