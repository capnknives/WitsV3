"""Tests for the math tool."""
import pytest
import numpy as np
from tools.math_tool import MathTool

@pytest.fixture
def math_tool():
    """Create a MathTool instance for testing."""
    return MathTool()

def test_basic_statistics(math_tool):
    """Test basic statistical calculations."""
    data = [1, 2, 3, 4, 5]
    result = math_tool.execute(
        operation="basic_stats",
        data=data
    )
    
    assert result["success"] is True
    assert result["count"] == 5
    assert result["mean"] == 3.0
    assert result["median"] == 3.0
    assert result["std_dev"] == pytest.approx(1.5811, rel=1e-4)
    assert result["variance"] == pytest.approx(2.5, rel=1e-4)
    assert result["min"] == 1
    assert result["max"] == 5
    assert result["range"] == 4
    assert result["q1"] == 2.0
    assert result["q3"] == 4.0

def test_regression_analysis(math_tool):
    """Test regression analysis."""
    x = [1, 2, 3, 4, 5]
    y = [2, 4, 5, 4, 5]
    
    result = math_tool.execute(
        operation="regression",
        x=x,
        y=y
    )
    
    assert result["success"] is True
    assert "slope" in result
    assert "intercept" in result
    assert "r_squared" in result
    assert "p_value" in result
    assert "std_err" in result
    assert "predictions" in result
    assert len(result["predictions"]) == len(x)

def test_probability_calculations(math_tool):
    """Test probability calculations."""
    # Test normal distribution
    result = math_tool.execute(
        operation="probability",
        distribution="normal",
        mean=0,
        std_dev=1,
        x=0
    )
    
    assert result["success"] is True
    assert "pdf" in result
    assert "cdf" in result
    assert result["pdf"] == pytest.approx(0.3989, rel=1e-4)
    assert result["cdf"] == pytest.approx(0.5, rel=1e-4)

    # Test binomial distribution
    result = math_tool.execute(
        operation="probability",
        distribution="binomial",
        n=10,
        p=0.5,
        k=5
    )
    
    assert result["success"] is True
    assert "pdf" in result
    assert "cdf" in result
    assert result["pdf"] == pytest.approx(0.2461, rel=1e-4)

def test_matrix_operations(math_tool):
    """Test matrix operations."""
    # Test matrix multiplication
    a = [[1, 2], [3, 4]]
    b = [[5, 6], [7, 8]]
    
    result = math_tool.execute(
        operation="matrix",
        operation_type="multiply",
        matrix_a=a,
        matrix_b=b
    )
    
    assert result["success"] is True
    assert result["result"] == [[19, 22], [43, 50]]

    # Test matrix inverse
    result = math_tool.execute(
        operation="matrix",
        operation_type="inverse",
        matrix_a=a
    )
    
    assert result["success"] is True
    expected_inverse = [[-2.0, 1.0], [1.5, -0.5]]
    assert np.allclose(result["result"], expected_inverse)

    # Test determinant
    result = math_tool.execute(
        operation="matrix",
        operation_type="determinant",
        matrix_a=a
    )
    
    assert result["success"] is True
    assert result["result"] == -2

def test_optimization(math_tool):
    """Test optimization calculations."""
    # Test linear programming
    c = [1, 1]  # Objective function coefficients
    A = [[1, 1], [2, 1]]  # Constraint coefficients
    b = [4, 5]  # Constraint bounds
    
    result = math_tool.execute(
        operation="optimization",
        method="linear",
        objective=c,
        constraints_a=A,
        constraints_b=b
    )
    
    assert result["success"] is True
    assert "optimal_value" in result
    assert "optimal_point" in result
    assert len(result["optimal_point"]) == len(c)

def test_math_schema(math_tool):
    """Test math tool schema."""
    schema = math_tool.get_schema()
    
    assert schema["name"] == "math"
    assert "operation" in schema["parameters"]["properties"]
    assert schema["parameters"]["required"] == ["operation"]

def test_error_handling(math_tool):
    """Test error handling in math operations."""
    # Test invalid operation
    result = math_tool.execute(
        operation="invalid_operation"
    )
    assert result["success"] is False
    assert "error" in result

    # Test invalid matrix dimensions
    result = math_tool.execute(
        operation="matrix",
        operation_type="multiply",
        matrix_a=[[1, 2]],
        matrix_b=[[1], [2], [3]]
    )
    assert result["success"] is False
    assert "error" in result

    # Test invalid probability parameters
    result = math_tool.execute(
        operation="probability",
        distribution="normal",
        mean=0,
        std_dev=-1,  # Invalid standard deviation
        x=0
    )
    assert result["success"] is False
    assert "error" in result 