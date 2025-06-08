Math Tool
========

The Math Tool provides comprehensive mathematical and statistical operations.

Overview
-------

The Math Tool offers a wide range of mathematical and statistical operations, including basic statistics, regression analysis, probability calculations, matrix operations, and optimization. It leverages NumPy and SciPy for efficient numerical computations.

Installation
----------

The Math Tool is included in the core WitsV3 package. No additional installation is required.

Usage
----

Basic Usage
~~~~~~~~~

.. code-block:: python

   from witsv3.tools import MathTool

   # Initialize the tool
   tool = MathTool()

   # Calculate basic statistics
   result = await tool.execute(
       operation="basic_stats",
       data=[1, 2, 3, 4, 5]
   )

Parameters
~~~~~~~~~

* **operation** (str, required): The operation to perform
  * "basic_stats": Calculate basic statistics
  * "regression": Perform regression analysis
  * "probability": Calculate probabilities
  * "matrix": Perform matrix operations
  * "optimization": Solve optimization problems

* **data** (list/array, required): The data to operate on
* **options** (dict, optional): Additional operation-specific options

Return Value
~~~~~~~~~~

The tool returns a dictionary with the following structure:

.. code-block:: python

   {
       "success": True,
       "result": {
           # Operation-specific results
       },
       "error": None
   }

Error Handling
~~~~~~~~~~~~

The tool handles various error cases:

* Invalid data types
* Insufficient data
* Numerical errors
* Matrix dimension errors
* Optimization failures

Example error response:

.. code-block:: python

   {
       "success": False,
       "result": None,
       "error": "Error message"
   }

Advanced Usage
-----------

Basic Statistics
~~~~~~~~~~~~~

Calculate basic statistics:

.. code-block:: python

   result = await tool.execute(
       operation="basic_stats",
       data=[1, 2, 3, 4, 5],
       options={
           "include_quartiles": True,
           "include_mode": True
       }
   )

Regression Analysis
~~~~~~~~~~~~~~~~

Perform linear regression:

.. code-block:: python

   result = await tool.execute(
       operation="regression",
       data={
           "x": [1, 2, 3, 4, 5],
           "y": [2, 4, 5, 4, 5]
       },
       options={
           "method": "linear",
           "include_predictions": True
       }
   )

Probability Calculations
~~~~~~~~~~~~~~~~~~~~~

Calculate probabilities:

.. code-block:: python

   result = await tool.execute(
       operation="probability",
       data={
           "distribution": "normal",
           "mean": 0,
           "std": 1
       },
       options={
           "x": 1.96,
           "calculate_cdf": True
       }
   )

Matrix Operations
~~~~~~~~~~~~~~

Perform matrix operations:

.. code-block:: python

   result = await tool.execute(
       operation="matrix",
       data={
           "a": [[1, 2], [3, 4]],
           "b": [[5, 6], [7, 8]]
       },
       options={
           "operation": "multiply"
       }
   )

Optimization
~~~~~~~~~~

Solve optimization problems:

.. code-block:: python

   result = await tool.execute(
       operation="optimization",
       data={
           "objective": "x**2 + y**2",
           "constraints": ["x + y = 1"]
       },
       options={
           "method": "SLSQP",
           "initial_guess": [0.5, 0.5]
       }
   )

Best Practices
-----------

1. **Data Validation**
   * Check data types
   * Validate dimensions
   * Handle missing values

2. **Numerical Stability**
   * Use appropriate methods
   * Handle edge cases
   * Check convergence

3. **Performance**
   * Use vectorized operations
   * Optimize algorithms
   * Cache results

4. **Error Handling**
   * Catch specific exceptions
   * Provide clear messages
   * Log error details

Example Use Cases
--------------

1. **Data Analysis**
   * Calculate statistics
   * Fit models
   * Make predictions

2. **Scientific Computing**
   * Solve equations
   * Optimize functions
   * Process signals

3. **Machine Learning**
   * Feature engineering
   * Model evaluation
   * Parameter tuning

API Reference
-----------

.. py:class:: MathTool

   Mathematical and statistical operations tool.

   .. py:method:: execute(operation: str, data: Union[list, dict], **kwargs) -> dict

      Execute a mathematical operation.

      :param operation: Operation to perform
      :param data: Data to operate on
      :param kwargs: Additional operation-specific parameters
      :return: Dictionary containing operation results

   .. py:method:: get_schema() -> dict

      Get the tool's schema for LLM consumption.

      :return: Dictionary containing tool schema

Limitations
---------

* Limited precision
* Memory constraints
* Algorithm limitations
* No symbolic math
* No complex numbers

Troubleshooting
------------

Common Issues
~~~~~~~~~~~

1. **Numerical Errors**
   * Check data range
   * Verify methods
   * Handle precision

2. **Memory Issues**
   * Reduce data size
   * Use efficient methods
   * Optimize algorithms

3. **Convergence Problems**
   * Check initial values
   * Verify constraints
   * Adjust parameters

Support
------

For issues and feature requests, please visit the `GitHub repository <https://github.com/yourusername/witsv3>`_. 