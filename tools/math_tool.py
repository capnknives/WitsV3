"""
Math and Statistics Tool for WitsV3.
Provides mathematical operations and statistical analysis.
"""

import logging
import math
import statistics
from typing import Any, Dict, List, Optional, Union
import numpy as np
from scipy import stats

from core.base_tool import BaseTool
from core.schemas import ToolCall

logger = logging.getLogger(__name__)

class MathTool(BaseTool):
    """Tool for mathematical operations and statistical analysis."""

    def __init__(self):
        """Initialize the math tool."""
        super().__init__(
            name="math_operations",
            description="Perform mathematical operations and statistical analysis on data."
        )

    async def execute(
        self,
        operation: str,
        data: Union[List[float], Dict[str, Any]],
        **kwargs: Any
    ) -> Dict[str, Any]:
        """
        Execute a mathematical operation.

        Args:
            operation: The operation to perform
            data: The data to operate on
            **kwargs: Additional operation-specific parameters

        Returns:
            Dictionary containing the operation result
        """
        try:
            if operation == "basic_stats":
                return await self._basic_statistics(data)
            elif operation == "regression":
                return await self._regression_analysis(data)
            elif operation == "probability":
                return await self._probability_calculations(data, **kwargs)
            elif operation == "matrix":
                return await self._matrix_operations(data, **kwargs)
            elif operation == "optimization":
                return await self._optimization(data, **kwargs)
            else:
                return {
                    "success": False,
                    "error": f"Unknown operation: {operation}"
                }

        except Exception as e:
            logger.error(f"Error in math operation: {e}")
            return {
                "success": False,
                "error": str(e)
            }

    async def _basic_statistics(self, data: List[float]) -> Dict[str, Any]:
        """Calculate basic statistics for a dataset."""
        try:
            if not data:
                return {
                    "success": False,
                    "error": "Empty dataset"
                }

            # Convert to numpy array for efficient calculations
            arr = np.array(data)

            # Calculate mode safely
            try:
                mode_result = stats.mode(arr, keepdims=True)
                mode_value = float(mode_result.mode[0])
            except:
                # If mode calculation fails, use the first value as fallback
                mode_value = float(arr[0])

            return {
                "success": True,
                "results": {
                    "count": len(data),
                    "mean": float(np.mean(arr)),
                    "median": float(np.median(arr)),
                    "mode": mode_value,
                    "std_dev": float(np.std(arr, ddof=1)),
                    "variance": float(np.var(arr, ddof=1)),
                    "min": float(np.min(arr)),
                    "max": float(np.max(arr)),
                    "range": float(np.max(arr) - np.min(arr)),
                    "quartiles": {
                        "q1": float(np.percentile(arr, 25)),
                        "q2": float(np.percentile(arr, 50)),
                        "q3": float(np.percentile(arr, 75))
                    }
                }
            }

        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }

    async def _regression_analysis(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Perform regression analysis."""
        try:
            if "x" not in data or "y" not in data:
                return {
                    "success": False,
                    "error": "Missing x or y data"
                }

            x = np.array(data["x"])
            y = np.array(data["y"])

            if len(x) != len(y):
                return {
                    "success": False,
                    "error": "x and y must have the same length"
                }

            # Linear regression
            slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)

            # Calculate predictions
            y_pred = slope * x + intercept

            # Calculate R-squared
            r_squared = r_value ** 2

            return {
                "success": True,
                "results": {
                    "slope": float(slope),
                    "intercept": float(intercept),
                    "r_squared": float(r_squared),
                    "p_value": float(p_value),
                    "std_err": float(std_err),
                    "predictions": y_pred.tolist()
                }
            }

        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }

    async def _probability_calculations(
        self,
        data: Dict[str, Any],
        **kwargs: Any
    ) -> Dict[str, Any]:
        """Perform probability calculations."""
        try:
            if "distribution" not in data:
                return {
                    "success": False,
                    "error": "Missing distribution type"
                }

            dist_type = data["distribution"]
            params = data.get("parameters", {})

            if dist_type == "normal":
                mean = params.get("mean", 0)
                std = params.get("std", 1)
                x = params.get("x")

                if x is None:
                    return {
                        "success": False,
                        "error": "Missing x value for normal distribution"
                    }

                # Validate parameters
                if std <= 0:
                    return {
                        "success": False,
                        "error": "Standard deviation must be positive"
                    }

                # Calculate probability density
                pdf = stats.norm.pdf(x, mean, std)
                # Calculate cumulative probability
                cdf = stats.norm.cdf(x, mean, std)

                return {
                    "success": True,
                    "results": {
                        "pdf": float(pdf),
                        "cdf": float(cdf)
                    }
                }

            elif dist_type == "binomial":
                n = params.get("n")
                p = params.get("p")
                k = params.get("k")

                if None in (n, p, k):
                    return {
                        "success": False,
                        "error": "Missing parameters for binomial distribution"
                    }

                # Validate parameters
                if not (0 <= p <= 1):
                    return {
                        "success": False,
                        "error": "Probability p must be between 0 and 1"
                    }
                if n <= 0 or k < 0 or k > n:
                    return {
                        "success": False,
                        "error": "Invalid parameters for binomial distribution"
                    }

                # Calculate probability mass
                pmf = stats.binom.pmf(k, n, p)
                # Calculate cumulative probability
                cdf = stats.binom.cdf(k, n, p)

                return {
                    "success": True,
                    "results": {
                        "pmf": float(pmf),
                        "cdf": float(cdf)
                    }
                }

            else:
                return {
                    "success": False,
                    "error": f"Unsupported distribution: {dist_type}"
                }

        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }

    async def _matrix_operations(
        self,
        data: Dict[str, Any],
        **kwargs: Any
    ) -> Dict[str, Any]:
        """Perform matrix operations."""
        try:
            if "operation" not in data or "matrices" not in data:
                return {
                    "success": False,
                    "error": "Missing operation or matrices"
                }

            operation = data["operation"]
            matrices = data["matrices"]

            if not isinstance(matrices, list) or len(matrices) < 1:
                return {
                    "success": False,
                    "error": "Invalid matrices data"
                }

            # Convert matrices to numpy arrays
            np_matrices = [np.array(m) for m in matrices]

            if operation == "multiply":
                if len(np_matrices) != 2:
                    return {
                        "success": False,
                        "error": "Matrix multiplication requires exactly 2 matrices"
                    }
                result = np.matmul(np_matrices[0], np_matrices[1])

            elif operation == "add":
                if len(np_matrices) != 2:
                    return {
                        "success": False,
                        "error": "Matrix addition requires exactly 2 matrices"
                    }
                result = np.add(np_matrices[0], np_matrices[1])

            elif operation == "inverse":
                if len(np_matrices) != 1:
                    return {
                        "success": False,
                        "error": "Matrix inverse requires exactly 1 matrix"
                    }
                result = np.linalg.inv(np_matrices[0])

            elif operation == "determinant":
                if len(np_matrices) != 1:
                    return {
                        "success": False,
                        "error": "Determinant calculation requires exactly 1 matrix"
                    }
                result = np.linalg.det(np_matrices[0])

            else:
                return {
                    "success": False,
                    "error": f"Unsupported matrix operation: {operation}"
                }

            return {
                "success": True,
                "result": result.tolist() if isinstance(result, np.ndarray) else float(result)
            }

        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }

    async def _optimization(
        self,
        data: Dict[str, Any],
        **kwargs: Any
    ) -> Dict[str, Any]:
        """Perform optimization calculations."""
        try:
            if "function" not in data or "method" not in data:
                return {
                    "success": False,
                    "error": "Missing function or method"
                }

            method = data["method"]
            func = data["function"]
            bounds = data.get("bounds")
            initial_guess = data.get("initial_guess")

            if method == "minimize":
                if not bounds and not initial_guess:
                    return {
                        "success": False,
                        "error": "Missing bounds or initial guess for minimization"
                    }

                if bounds:
                    result = stats.minimize(
                        func,
                        x0=initial_guess if initial_guess else None,
                        bounds=bounds
                    )
                else:
                    result = stats.minimize(
                        func,
                        x0=initial_guess
                    )

                return {
                    "success": True,
                    "results": {
                        "x": result.x.tolist(),
                        "fun": float(result.fun),
                        "success": bool(result.success),
                        "message": str(result.message)
                    }
                }

            else:
                return {
                    "success": False,
                    "error": f"Unsupported optimization method: {method}"
                }

        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }

    def get_schema(self) -> Dict[str, Any]:
        """Get the tool's schema for LLM consumption."""
        return {
            "name": "math_operations",
            "description": "Perform mathematical operations and statistical analysis",
            "parameters": {
                "type": "object",
                "properties": {
                    "operation": {
                        "type": "string",
                        "description": "The operation to perform",
                        "enum": ["basic_stats", "regression", "probability", "matrix", "optimization"]
                    },
                    "data": {
                        "type": ["array", "object"],
                        "description": "The data to operate on"
                    }
                },
                "required": ["operation", "data"]
            }
        }
