---
title: "Backlog Clearance Plan: Phase 1 Implementation"
created: "2025-06-11"
last_updated: "2025-06-11"
status: "active"
---

# Backlog Clearance Plan: Phase 1 Implementation

This document outlines the detailed implementation plan for Phase 1 of the WitsV3 backlog clearance, focusing on critical fixes that need to be addressed immediately.

## Overview

Phase 1 focuses on addressing critical issues that affect system stability and reliability. These fixes are prioritized to ensure a solid foundation for subsequent phases of development.

## Timeline

- **Start Date**: June 12, 2025
- **End Date**: June 15, 2025
- **Duration**: 4 days

## Tasks

### 1. Fix Memory Pruning Issue

**Description**: Implement automatic pruning in the MemoryManager to prevent memory files from growing too large.

**Implementation Steps**:

1. **Update MemoryManager Configuration**
   ```python
   class MemoryManagerSettings(BaseModel):
       # Existing settings...

       # Memory pruning settings
       enable_auto_pruning: bool = Field(default=True)
       max_memory_size_mb: int = Field(default=100, gt=0)
       pruning_threshold: float = Field(default=0.8, ge=0.0, le=1.0)
       pruning_strategy: str = Field(default="oldest_first",
                                    pattern="^(oldest_first|least_relevant|hybrid)$")
   ```

2. **Implement Pruning Logic in MemoryManager**
   ```python
   async def check_and_prune_memory(self) -> None:
       """
       Check memory size and prune if necessary.

       Prunes memory segments based on the configured strategy when
       the memory size exceeds the threshold.
       """
       if not self.config.memory_manager.enable_auto_pruning:
           return

       current_size = await self._get_memory_size_mb()
       max_size = self.config.memory_manager.max_memory_size_mb
       threshold = self.config.memory_manager.pruning_threshold

       if current_size >= (max_size * threshold):
           self.logger.info(f"Memory size ({current_size}MB) exceeds threshold "
                           f"({max_size * threshold}MB). Pruning...")
           await self._prune_memory()
   ```

3. **Add Size Monitoring**
   ```python
   async def _get_memory_size_mb(self) -> float:
       """Get the current memory size in megabytes."""
       # Implementation depends on the backend
       if isinstance(self.backend, JSONMemoryBackend):
           # For JSON backend, check file size
           memory_file = self.config.memory_manager.memory_file
           if os.path.exists(memory_file):
               size_bytes = os.path.getsize(memory_file)
               return size_bytes / (1024 * 1024)  # Convert to MB
       elif isinstance(self.backend, FaissMemoryBackend):
           # For FAISS backend, estimate size based on number of vectors
           # and vector dimension
           return await self.backend.estimate_size_mb()
       # Add other backend types as needed
       return 0.0
   ```

4. **Implement Pruning Strategies**
   ```python
   async def _prune_memory(self) -> None:
       """Prune memory based on the configured strategy."""
       strategy = self.config.memory_manager.pruning_strategy

       if strategy == "oldest_first":
           await self._prune_oldest_segments()
       elif strategy == "least_relevant":
           await self._prune_least_relevant_segments()
       elif strategy == "hybrid":
           await self._prune_hybrid()
   ```

5. **Add Pruning to Background Tasks**
   - Update the BackgroundAgent to periodically check and prune memory
   - Add pruning to the scheduled tasks

**Testing**:
- Create unit tests for each pruning strategy
- Test with different memory backends
- Verify memory size stays below threshold

### 2. Implement Tool Argument Validation

**Description**: Add Pydantic validation for tool arguments to prevent runtime errors from invalid inputs.

**Implementation Steps**:

1. **Enhance BaseTool with Validation**
   ```python
   class BaseTool(ABC):
       # Existing code...

       async def validate_args(self, args: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
           """
           Validate tool arguments against the schema.

           Args:
               args: The arguments to validate

           Returns:
               Tuple of (is_valid, error_message)
           """
           try:
               # Get the schema for this tool
               schema = self.get_schema()
               if not schema:
                   return True, None

               # Create a Pydantic model dynamically from the schema
               model = create_model('ToolArgs', **{
                   k: (self._get_type_from_schema(v), ... if v.get('required', False) else None)
                   for k, v in schema.get('properties', {}).items()
               })

               # Validate the arguments
               model(**args)
               return True, None
           except ValidationError as e:
               return False, str(e)

       def _get_type_from_schema(self, property_schema: Dict[str, Any]) -> Type:
           """Convert JSON schema type to Python type."""
           type_map = {
               'string': str,
               'integer': int,
               'number': float,
               'boolean': bool,
               'array': list,
               'object': dict
           }
           return type_map.get(property_schema.get('type', 'string'), str)
   ```

2. **Add Pre-execution Validation Hook**
   ```python
   async def execute(self, **kwargs) -> ToolResult:
       """
       Execute the tool with the provided arguments.

       Args:
           **kwargs: Tool arguments

       Returns:
           ToolResult with the execution result
       """
       # Validate arguments before execution
       is_valid, error_message = await self.validate_args(kwargs)
       if not is_valid:
           return ToolResult(
               success=False,
               error=f"Invalid arguments: {error_message}",
               result=None
           )

       # Continue with existing execution logic
       try:
           # Existing execution code...
       except Exception as e:
           # Existing error handling...
   ```

3. **Add Helpful Error Messages**
   - Enhance error messages to include expected types and constraints
   - Add examples of valid arguments in error messages

4. **Update ToolRegistry to Support Validation**
   - Ensure ToolRegistry passes the validation results to the LLM
   - Add validation information to tool descriptions

**Testing**:
- Create tests with valid and invalid arguments for each tool
- Verify error messages are clear and helpful
- Test with different argument types and edge cases

### 3. Enhance Error Context in Streaming Responses

**Description**: Improve error handling in StreamData to provide more context in error messages.

**Implementation Steps**:

1. **Enhance StreamData Error Type**
   ```python
   class StreamData:
       # Existing code...

       @classmethod
       def stream_error(cls,
                       message: str,
                       error_type: Optional[str] = None,
                       context: Optional[Dict[str, Any]] = None,
                       traceback: Optional[str] = None) -> 'StreamData':
           """
           Create a stream error with enhanced context.

           Args:
               message: The error message
               error_type: Type of error (e.g., "ValidationError", "NetworkError")
               context: Additional context about the error
               traceback: Optional traceback information

           Returns:
               StreamData object with error information
           """
           return cls(
               type="error",
               content={
                   "message": message,
                   "error_type": error_type or "GeneralError",
                   "context": context or {},
                   "traceback": traceback,
                   "timestamp": datetime.now().isoformat()
               }
           )
   ```

2. **Implement Error Tracing Across Components**
   ```python
   class ErrorContext:
       """Helper class to track error context across components."""

       @staticmethod
       def capture_exception(e: Exception) -> Dict[str, Any]:
           """Capture exception details."""
           import traceback
           return {
               "error_type": type(e).__name__,
               "message": str(e),
               "traceback": traceback.format_exc(),
               "timestamp": datetime.now().isoformat()
           }

       @staticmethod
       def from_exception(e: Exception,
                         component: str,
                         operation: str,
                         additional_context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
           """Create error context from an exception."""
           context = ErrorContext.capture_exception(e)
           context.update({
               "component": component,
               "operation": operation
           })
           if additional_context:
               context.update(additional_context)
           return context
   ```

3. **Update Tools to Use Enhanced Error Context**
   ```python
   async def execute(self, **kwargs) -> ToolResult:
       try:
           # Existing execution code...
       except Exception as e:
           error_context = ErrorContext.from_exception(
               e,
               component=self.__class__.__name__,
               operation="execute",
               additional_context={"arguments": kwargs}
           )
           return ToolResult(
               success=False,
               error=f"Error executing {self.name}: {str(e)}",
               result=None,
               error_context=error_context
           )
   ```

4. **Update Response Parser to Handle Error Context**
   - Modify ResponseParser to include error context in responses
   - Ensure error context is properly formatted for the LLM

**Testing**:
- Create tests that trigger different types of errors
- Verify error context is properly captured and formatted
- Test error propagation across components

### 4. Fix Gemma Model Crashes

**Description**: Implement robust error handling for Gemma model failures and add automatic fallback to alternative models.

**Implementation Steps**:

1. **Add Model Fallback Configuration**
   ```python
   class LLMSettings(BaseModel):
       # Existing settings...

       # Model fallback settings
       enable_model_fallback: bool = Field(default=True)
       fallback_models: List[str] = Field(default=["llama3"])
       max_fallback_attempts: int = Field(default=3, ge=1)
   ```

2. **Implement Robust Error Handling**
   ```python
   async def generate_completion(self,
                               prompt: str,
                               model: Optional[str] = None,
                               **kwargs) -> AsyncGenerator[str, None]:
       """
       Generate a completion with robust error handling and fallback.

       Args:
           prompt: The prompt to send to the LLM
           model: The model to use (optional)
           **kwargs: Additional parameters

       Yields:
           Generated text chunks
       """
       selected_model = model or self.config.llm.default_model
       attempts = 0
       fallback_models = [selected_model] + self.config.llm.fallback_models

       while attempts < len(fallback_models) and attempts < self.config.llm.max_fallback_attempts:
           current_model = fallback_models[attempts]
           try:
               self.logger.info(f"Generating completion with model: {current_model}")
               async for chunk in self._generate_completion_internal(prompt, current_model, **kwargs):
                   yield chunk
               break  # Success, exit the loop
           except Exception as e:
               attempts += 1
               error_context = ErrorContext.from_exception(
                   e,
                   component="LLMInterface",
                   operation="generate_completion",
                   additional_context={"model": current_model, "attempt": attempts}
               )
               self.logger.error(f"Error with model {current_model}: {str(e)}")

               if attempts >= len(fallback_models) or attempts >= self.config.llm.max_fallback_attempts:
                   self.logger.error("All fallback attempts failed")
                   yield json.dumps({
                       "error": f"All model attempts failed: {str(e)}",
                       "context": error_context
                   })
                   break

               self.logger.info(f"Falling back to model: {fallback_models[attempts]}")
   ```

3. **Add Comprehensive Logging**
   ```python
   class ModelLogger:
       """Enhanced logging for model operations."""

       def __init__(self, logger: logging.Logger):
           self.logger = logger

       def log_model_error(self,
                          model: str,
                          error: Exception,
                          context: Dict[str, Any]) -> None:
           """Log detailed model error information."""
           error_id = str(uuid.uuid4())[:8]
           self.logger.error(f"Model error [{error_id}] with {model}: {str(error)}")

           # Log detailed error information at debug level
           debug_info = {
               "error_id": error_id,
               "model": model,
               "error_type": type(error).__name__,
               "error_message": str(error),
               "context": context,
               "timestamp": datetime.now().isoformat()
           }
           self.logger.debug(f"Model error details: {json.dumps(debug_info, indent=2)}")

           # Also log to a dedicated model errors file
           self._log_to_file(debug_info)

       def _log_to_file(self, debug_info: Dict[str, Any]) -> None:
           """Log error details to a dedicated file."""
           try:
               log_dir = "logs/model_errors"
               os.makedirs(log_dir, exist_ok=True)

               log_file = f"{log_dir}/model_errors_{datetime.now().strftime('%Y%m%d')}.jsonl"
               with open(log_file, "a") as f:
                   f.write(json.dumps(debug_info) + "\n")
           except Exception as e:
               self.logger.error(f"Failed to write to model error log: {str(e)}")
   ```

4. **Update LLMInterface to Use ModelLogger**
   - Integrate ModelLogger with LLMInterface
   - Add detailed logging for model operations

**Testing**:
- Create tests that simulate model failures
- Verify fallback mechanism works correctly
- Test with different fallback configurations
- Verify logging captures all necessary information

## Implementation Schedule

| Task | Start Date | End Date | Owner |
|------|------------|----------|-------|
| Fix memory pruning issue | June 12, 2025 | June 13, 2025 | TBD |
| Implement tool argument validation | June 12, 2025 | June 14, 2025 | TBD |
| Enhance error context in streaming responses | June 13, 2025 | June 14, 2025 | TBD |
| Fix Gemma model crashes | June 14, 2025 | June 15, 2025 | TBD |

## Success Criteria

- All memory files stay below the configured size threshold
- Tool execution fails gracefully with clear error messages for invalid arguments
- Error messages include detailed context for debugging
- Gemma model failures automatically fall back to alternative models
- All tests pass with the new implementations

## Dependencies

- Existing MemoryManager implementation
- BaseTool and ToolRegistry classes
- StreamData implementation
- LLMInterface implementation

## Risks and Mitigations

| Risk | Mitigation |
|------|------------|
| Memory pruning could remove important information | Implement configurable pruning strategies and retention policies |
| Validation could be too strict and reject valid inputs | Add flexible validation with clear error messages and examples |
| Error context could expose sensitive information | Implement filtering for sensitive data in error contexts |
| Model fallback might not work for all use cases | Add configuration options for fallback behavior and model selection |

## Next Steps

After completing Phase 1, the team will:

1. Update documentation with the new features and fixes
2. Run comprehensive tests to verify stability
3. Begin work on Phase 2: Complete Neural Web
