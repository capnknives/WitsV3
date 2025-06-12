---
title: "Backlog Clearance Plan: Phase 4 Implementation"
created: "2025-06-11"
last_updated: "2025-06-11"
status: "active"
---

# Backlog Clearance Plan: Phase 4 Implementation

This document outlines the detailed implementation plan for Phase 4 of the WitsV3 backlog clearance, focusing on New Features to expand the system's capabilities.

## Overview

Phase 4 builds on the foundation established in the previous phases by adding new features to the WitsV3 system. This phase focuses on implementing a Web UI prototype, adding Langchain integration, and enhancing background agent monitoring.

## Timeline

- **Start Date**: June 26, 2025
- **End Date**: July 9, 2025
- **Duration**: 14 days

## Tasks

### 1. Web UI Prototype

**Description**: Create a basic web interface for WitsV3 using FastAPI and React.

**Implementation Steps**:

1. **Create FastAPI Backend**
   ```python
   # In gui/api/main.py
   from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Depends, HTTPException, status
   from fastapi.middleware.cors import CORSMiddleware
   from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
   from pydantic import BaseModel
   from typing import List, Dict, Any, Optional
   import asyncio
   import json
   import logging
   import os
   import sys

   # Add parent directory to path to import WitsV3 modules
   sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

   from core.config import WitsV3Config
   from core.llm_interface import LLMInterface
   from core.memory_manager import MemoryManager
   from core.tool_registry import ToolRegistry
   from agents.wits_control_center_agent import WitsControlCenterAgent
   from core.schemas import StreamData

   # Initialize FastAPI app
   app = FastAPI(title="WitsV3 API", version="0.1.0")

   # Add CORS middleware
   app.add_middleware(
       CORSMiddleware,
       allow_origins=["*"],  # In production, restrict this to specific origins
       allow_credentials=True,
       allow_methods=["*"],
       allow_headers=["*"],
   )

   # Initialize OAuth2 for authentication
   oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

   # Initialize logging
   logging.basicConfig(level=logging.INFO)
   logger = logging.getLogger(__name__)

   # Initialize WitsV3 components
   config = WitsV3Config()
   llm_interface = LLMInterface(config)
   memory_manager = MemoryManager(config)
   tool_registry = ToolRegistry(config)

   # Define Pydantic models for API
   class ChatRequest(BaseModel):
       message: str
       user_id: str
       session_id: Optional[str] = None

   class ChatResponse(BaseModel):
       response: str
       thinking: Optional[str] = None
       actions: Optional[List[Dict[str, Any]]] = None

   class TokenRequest(BaseModel):
       username: str
       password: str

   class TokenResponse(BaseModel):
       access_token: str
       token_type: str

   # Active WebSocket connections
   active_connections: Dict[str, WebSocket] = {}

   # Authentication functions
   async def authenticate_user(username: str, password: str) -> bool:
       # In a real implementation, this would check against a database
       # For now, use a simple hardcoded check
       return username == "admin" and password == "password"

   async def get_current_user(token: str = Depends(oauth2_scheme)):
       # In a real implementation, this would validate the token
       # For now, just return the token as the user ID
       return token

   # API routes
   @app.post("/token", response_model=TokenResponse)
   async def login_for_access_token(form_data: OAuth2PasswordRequestForm = Depends()):
       user = await authenticate_user(form_data.username, form_data.password)
       if not user:
           raise HTTPException(
               status_code=status.HTTP_401_UNAUTHORIZED,
               detail="Incorrect username or password",
               headers={"WWW-Authenticate": "Bearer"},
           )
       # In a real implementation, this would generate a JWT token
       access_token = form_data.username
       return {"access_token": access_token, "token_type": "bearer"}

   @app.post("/chat", response_model=ChatResponse)
   async def chat(request: ChatRequest, current_user: str = Depends(get_current_user)):
       # Create a control center agent
       agent = WitsControlCenterAgent(config)

       # Process the message
       response_parts = {
           "response": "",
           "thinking": "",
           "actions": []
       }

       async for stream_data in agent.process_message(request.message, request.user_id):
           if stream_data.type == "thinking":
               response_parts["thinking"] += stream_data.content
           elif stream_data.type == "action":
               response_parts["actions"].append(stream_data.content)
           elif stream_data.type == "response":
               response_parts["response"] += stream_data.content

       return ChatResponse(**response_parts)

   @app.websocket("/ws/{client_id}")
   async def websocket_endpoint(websocket: WebSocket, client_id: str):
       await websocket.accept()
       active_connections[client_id] = websocket

       try:
           while True:
               # Receive message from client
               data = await websocket.receive_text()
               request = json.loads(data)

               # Create a control center agent
               agent = WitsControlCenterAgent(config)

               # Process the message
               async for stream_data in agent.process_message(request["message"], request["user_id"]):
                   # Send stream data to client
                   await websocket.send_json({
                       "type": stream_data.type,
                       "content": stream_data.content
                   })
       except WebSocketDisconnect:
           # Remove connection when client disconnects
           if client_id in active_connections:
               del active_connections[client_id]

   @app.get("/tools")
   async def get_tools(current_user: str = Depends(get_current_user)):
       # Get all registered tools
       tools = tool_registry.get_all_tools()
       return {"tools": [tool.get_llm_description() for tool in tools]}

   @app.get("/models")
   async def get_models(current_user: str = Depends(get_current_user)):
       # Get available models
       models = await llm_interface.get_available_models()
       return {"models": models}

   # Run the FastAPI app
   if __name__ == "__main__":
       import uvicorn
       uvicorn.run(app, host="0.0.0.0", port=8000)
   ```

2. **Implement React Frontend**
   ```javascript
   // In gui/web/src/App.js
   import React, { useState, useEffect, useRef } from 'react';
   import { ChakraProvider, Box, VStack, HStack, Input, Button, Text, Heading, Spinner, useToast } from '@chakra-ui/react';
   import { ChatIcon, SettingsIcon } from '@chakra-ui/icons';
   import './App.css';

   function App() {
     const [messages, setMessages] = useState([]);
     const [input, setInput] = useState('');
     const [isLoading, setIsLoading] = useState(false);
     const [isConnected, setIsConnected] = useState(false);
     const [thinking, setThinking] = useState('');
     const [actions, setActions] = useState([]);
     const [userId, setUserId] = useState('user-' + Math.random().toString(36).substring(2, 9));
     const [token, setToken] = useState(localStorage.getItem('token') || '');
     const [isLoggedIn, setIsLoggedIn] = useState(!!token);
     const [username, setUsername] = useState('');
     const [password, setPassword] = useState('');

     const toast = useToast();
     const messagesEndRef = useRef(null);
     const ws = useRef(null);

     // Connect to WebSocket
     useEffect(() => {
       if (isLoggedIn && !ws.current) {
         connectWebSocket();
       }

       return () => {
         if (ws.current) {
           ws.current.close();
         }
       };
     }, [isLoggedIn]);

     // Scroll to bottom of messages
     useEffect(() => {
       messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
     }, [messages]);

     const connectWebSocket = () => {
       const clientId = 'client-' + Math.random().toString(36).substring(2, 9);
       ws.current = new WebSocket(`ws://localhost:8000/ws/${clientId}`);

       ws.current.onopen = () => {
         setIsConnected(true);
         toast({
           title: 'Connected',
           description: 'WebSocket connection established',
           status: 'success',
           duration: 3000,
           isClosable: true,
         });
       };

       ws.current.onmessage = (event) => {
         const data = JSON.parse(event.data);

         if (data.type === 'thinking') {
           setThinking(prev => prev + data.content);
         } else if (data.type === 'action') {
           setActions(prev => [...prev, data.content]);
         } else if (data.type === 'response') {
           setMessages(prev => [
             ...prev,
             { role: 'assistant', content: data.content }
           ]);
           setIsLoading(false);
         }
       };

       ws.current.onclose = () => {
         setIsConnected(false);
         toast({
           title: 'Disconnected',
           description: 'WebSocket connection closed',
           status: 'warning',
           duration: 3000,
           isClosable: true,
         });
       };

       ws.current.onerror = (error) => {
         console.error('WebSocket error:', error);
         toast({
           title: 'Connection Error',
           description: 'Error connecting to server',
           status: 'error',
           duration: 3000,
           isClosable: true,
         });
       };
     };

     const handleLogin = async () => {
       try {
         const response = await fetch('http://localhost:8000/token', {
           method: 'POST',
           headers: {
             'Content-Type': 'application/x-www-form-urlencoded',
           },
           body: new URLSearchParams({
             'username': username,
             'password': password,
           }),
         });

         if (response.ok) {
           const data = await response.json();
           setToken(data.access_token);
           localStorage.setItem('token', data.access_token);
           setIsLoggedIn(true);

           toast({
             title: 'Login Successful',
             description: 'You are now logged in',
             status: 'success',
             duration: 3000,
             isClosable: true,
           });
         } else {
           toast({
             title: 'Login Failed',
             description: 'Invalid username or password',
             status: 'error',
             duration: 3000,
             isClosable: true,
           });
         }
       } catch (error) {
         console.error('Login error:', error);
         toast({
           title: 'Login Error',
           description: 'Error connecting to server',
           status: 'error',
           duration: 3000,
           isClosable: true,
         });
       }
     };

     const handleLogout = () => {
       setToken('');
       localStorage.removeItem('token');
       setIsLoggedIn(false);

       if (ws.current) {
         ws.current.close();
         ws.current = null;
       }

       toast({
         title: 'Logged Out',
         description: 'You have been logged out',
         status: 'info',
         duration: 3000,
         isClosable: true,
       });
     };

     const handleSendMessage = () => {
       if (!input.trim()) return;

       // Add user message to chat
       setMessages(prev => [...prev, { role: 'user', content: input }]);

       // Clear thinking and actions
       setThinking('');
       setActions([]);

       // Set loading state
       setIsLoading(true);

       // Send message via WebSocket if connected
       if (ws.current && ws.current.readyState === WebSocket.OPEN) {
         ws.current.send(JSON.stringify({
           message: input,
           user_id: userId,
         }));
       } else {
         // Fallback to REST API
         fetch('http://localhost:8000/chat', {
           method: 'POST',
           headers: {
             'Content-Type': 'application/json',
             'Authorization': `Bearer ${token}`,
           },
           body: JSON.stringify({
             message: input,
             user_id: userId,
           }),
         })
           .then(response => response.json())
           .then(data => {
             setMessages(prev => [
               ...prev,
               { role: 'assistant', content: data.response }
             ]);
             setThinking(data.thinking || '');
             setActions(data.actions || []);
             setIsLoading(false);
           })
           .catch(error => {
             console.error('Error:', error);
             setIsLoading(false);
             toast({
               title: 'Error',
               description: 'Failed to send message',
               status: 'error',
               duration: 3000,
               isClosable: true,
             });
           });
       }

       // Clear input
       setInput('');
     };

     // Login form
     if (!isLoggedIn) {
       return (
         <ChakraProvider>
           <Box p={8} maxWidth="500px" mx="auto" mt={20} borderWidth={1} borderRadius={8} boxShadow="lg">
             <VStack spacing={4} align="flex-start">
               <Heading>Login to WitsV3</Heading>
               <Input
                 placeholder="Username"
                 value={username}
                 onChange={(e) => setUsername(e.target.value)}
               />
               <Input
                 type="password"
                 placeholder="Password"
                 value={password}
                 onChange={(e) => setPassword(e.target.value)}
               />
               <Button colorScheme="blue" onClick={handleLogin} width="100%">
                 Login
               </Button>
             </VStack>
           </Box>
         </ChakraProvider>
       );
     }

     return (
       <ChakraProvider>
         <Box height="100vh" display="flex" flexDirection="column">
           {/* Header */}
           <Box p={4} bg="blue.500" color="white">
             <HStack justify="space-between">
               <Heading size="md">WitsV3 Web Interface</Heading>
               <HStack>
                 <Text>{isConnected ? 'Connected' : 'Disconnected'}</Text>
                 <Button size="sm" onClick={handleLogout}>Logout</Button>
               </HStack>
             </HStack>
           </Box>

           {/* Main content */}
           <Box flex="1" p={4} overflowY="auto">
             <VStack spacing={4} align="stretch">
               {/* Messages */}
               {messages.map((message, index) => (
                 <Box
                   key={index}
                   bg={message.role === 'user' ? 'blue.100' : 'gray.100'}
                   p={3}
                   borderRadius="md"
                   alignSelf={message.role === 'user' ? 'flex-end' : 'flex-start'}
                   maxWidth="70%"
                 >
                   <Text>{message.content}</Text>
                 </Box>
               ))}

               {/* Thinking */}
               {thinking && (
                 <Box bg="purple.100" p={3} borderRadius="md">
                   <Heading size="xs" mb={2}>Thinking:</Heading>
                   <Text fontFamily="monospace" fontSize="sm">{thinking}</Text>
                 </Box>
               )}

               {/* Actions */}
               {actions.length > 0 && (
                 <Box bg="green.100" p={3} borderRadius="md">
                   <Heading size="xs" mb={2}>Actions:</Heading>
                   <VStack align="stretch">
                     {actions.map((action, index) => (
                       <Box key={index} bg="white" p={2} borderRadius="sm">
                         <Text fontFamily="monospace" fontSize="sm">
                           {JSON.stringify(action, null, 2)}
                         </Text>
                       </Box>
                     ))}
                   </VStack>
                 </Box>
               )}

               {/* Loading indicator */}
               {isLoading && (
                 <Box alignSelf="flex-start">
                   <Spinner size="sm" mr={2} />
                   <Text display="inline">WitsV3 is thinking...</Text>
                 </Box>
               )}

               {/* Scroll anchor */}
               <div ref={messagesEndRef} />
             </VStack>
           </Box>

           {/* Input area */}
           <Box p={4} borderTopWidth={1}>
             <HStack>
               <Input
                 placeholder="Type your message..."
                 value={input}
                 onChange={(e) => setInput(e.target.value)}
                 onKeyPress={(e) => e.key === 'Enter' && handleSendMessage()}
               />
               <Button
                 colorScheme="blue"
                 onClick={handleSendMessage}
                 isLoading={isLoading}
                 leftIcon={<ChatIcon />}
               >
                 Send
               </Button>
             </HStack>
           </Box>
         </Box>
       </ChakraProvider>
     );
   }

   export default App;
   ```

3. **Add WebSocket for Streaming**
   ```python
   # In gui/api/websocket_manager.py
   from fastapi import WebSocket
   from typing import Dict, List, Any, Optional
   import logging
   import json

   class ConnectionManager:
       """Manage WebSocket connections."""

       def __init__(self):
           self.active_connections: Dict[str, WebSocket] = {}
           self.logger = logging.getLogger(__name__)

       async def connect(self, websocket: WebSocket, client_id: str) -> None:
           """Connect a client."""
           await websocket.accept()
           self.active_connections[client_id] = websocket
           self.logger.info(f"Client {client_id} connected. Total connections: {len(self.active_connections)}")

       def disconnect(self, client_id: str) -> None:
           """Disconnect a client."""
           if client_id in self.active_connections:
               del self.active_connections[client_id]
               self.logger.info(f"Client {client_id} disconnected. Total connections: {len(self.active_connections)}")

       async def send_json(self, client_id: str, data: Any) -> None:
           """Send JSON data to a client."""
           if client_id in self.active_connections:
               await self.active_connections[client_id].send_json(data)

       async def broadcast_json(self, data: Any) -> None:
           """Broadcast JSON data to all clients."""
           for connection in self.active_connections.values():
               await connection.send_json(data)

       async def send_stream_data(self, client_id: str, stream_data: Any) -> None:
           """Send stream data to a client."""
           if client_id in self.active_connections:
               await self.active_connections[client_id].send_json({
                   "type": stream_data.type,
                   "content": stream_data.content
               })
   ```

4. **Create API Documentation**
   ```python
   # In gui/api/docs.py
   from fastapi.openapi.docs import get_swagger_ui_html
   from fastapi import FastAPI

   def setup_docs(app: FastAPI) -> None:
       """Set up API documentation."""
       @app.get("/docs", include_in_schema=False)
       async def custom_swagger_ui_html():
           return get_swagger_ui_html(
               openapi_url=app.openapi_url,
               title=app.title + " - API Documentation",
               oauth2_redirect_url=app.swagger_ui_oauth2_redirect_url,
               swagger_js_url="https://cdn.jsdelivr.net/npm/swagger-ui-dist@4/swagger-ui-bundle.js",
               swagger_css_url="https://cdn.jsdelivr.net/npm/swagger-ui-dist@4/swagger-ui.css",
           )
   ```

### 2. Langchain Integration

**Description**: Create a bridge to integrate Langchain tools and capabilities with WitsV3.

**Implementation Steps**:

1. **Create Langchain Bridge**
   ```python
   # In core/langchain_bridge.py
   from typing import Dict, List, Any, Optional, Callable, Union, Type
   import logging
   import asyncio
   from pydantic import BaseModel, create_model

   # Import Langchain components
   from langchain.agents import Tool as LangchainTool
   from langchain.schema import BaseOutputParser
   from langchain.prompts import PromptTemplate
   from langchain.chains import LLMChain
   from langchain.llms.base import BaseLLM

   # Import WitsV3 components
   from core.base_tool import BaseTool
   from core.schemas import ToolResult
   from core.config import WitsV3Config

   class LangchainBridge:
       """Bridge between WitsV3 and Langchain."""

       def __init__(self, config: WitsV3Config):
           self.config = config
           self.logger = logging.getLogger(__name__)

       def langchain_tool_to_wits_tool(self, lc_tool: LangchainTool) -> BaseTool:
           """
           Convert a Langchain tool to a WitsV3 tool.

           Args:
               lc_tool: Langchain tool to convert

           Returns:
               WitsV3 tool
           """
           # Create a WitsV3 tool that wraps the Langchain tool
           class LangchainToolWrapper(BaseTool):
               def __init__(self, lc_tool: LangchainTool, config: WitsV3Config):
                   super().__init__(config)
                   self.lc_tool = lc_tool
                   self.name = lc_tool.name
                   self.description = lc_tool.description

               def get_schema(self) -> Dict[str, Any]:
                   # Create a simple schema with a single input parameter
                   return {
                       "type": "object",
                       "properties": {
                           "input": {
                               "type": "string",
                               "description": "Input for the tool"
                           }
                       },
                       "required": ["input"]
                   }

               async def execute(self, **kwargs) -> ToolResult:
                   try:
                       # Get input parameter
                       input_str = kwargs.get("input", "")

                       # Execute the Langchain tool
                       # If the tool is synchronous, run it in a thread pool
                       if asyncio.iscoroutinefunction(self.lc_tool._run):
                           result = await self.lc_tool._run(input_str)
                       else:
                           loop = asyncio.get_event_loop()
                           result = await loop.run_in_executor(
                               None, self.lc_tool._run, input_str
                           )

                       return ToolResult(
                           success=True,
                           result=result
                       )
                   except Exception as e:
                       return ToolResult(
                           success=False,
                           error=f"Error executing Langchain tool: {str(e)}"
                       )

           # Create and return the wrapper
           return LangchainToolWrapper(lc_tool, self.config)

       def wits_tool_to_langchain_tool(self, wits_tool: BaseTool) -> LangchainTool:
           """
           Convert a WitsV3 tool to a Langchain tool.

           Args:
               wits_tool: WitsV3 tool to convert

           Returns:
               Langchain tool
           """
           # Create a function that executes the WitsV3 tool
           async def run_wits_tool(input_str: str) -> str:
               # Execute the WitsV3 tool
               result = await wits_tool.execute(input=input_str)

               # Return the result as a string
               if result.success:
                   return str(result.result)
               else:
                   return f"Error: {result.error}"

           # Create a synchronous wrapper for the async function
           def run_wits_tool_sync(input_str: str) -> str:
               # Run the async function in a new event loop
               loop = asyncio.new_event_loop()
               try:
                   return loop.run_until_complete(run_wits_tool(input_str))
               finally:
                   loop.close()

           # Create and return the Langchain tool
           return LangchainTool(
               name=wits_tool.name,
               description=wits_tool.description,
               func=run_wits_tool_sync
           )

       async def run_langchain_chain(self,
                                  chain: Any,
                                  inputs: Dict[str, Any]) -> Dict[str, Any]:
           """
           Run a Langchain chain.

           Args:
               chain: Langchain chain to run
               inputs: Inputs for the chain

           Returns:
               Chain outputs
           """
           # Check if the chain has an async version of the call method
           if hasattr(chain, "acall") and asyncio.iscoroutinefunction(chain.acall):
               # Run the chain asynchronously
               return await chain.acall(inputs)
           else:
               # Run the chain synchronously in a thread pool
               loop = asyncio.get_event_loop()
               return await loop.run_in_executor(
                   None, chain, inputs
               )

       def create_langchain_llm_wrapper(self, model: Optional[str] = None) -> BaseLLM:
           """
           Create a Langchain LLM wrapper for WitsV3's LLM interface.

           Args:
               model: Optional model name to use

           Returns:
               Langchain LLM wrapper
           """
           from langchain.llms.base import LLM
           from core.llm_interface import LLMInterface

           # Create a Langchain LLM that uses WitsV3's LLM interface
           class WitsV3LLM(LLM):
               wits_config: WitsV3Config
               wits_model: Optional[str] = None

               @property
               def _llm_type(self) -> str:
                   return "witsv3"

               def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
                   """Run the LLM on the given prompt and input."""
                   # Create LLM interface
                   llm_interface = LLMInterface(self.wits_config)

                   # Run the LLM
                   loop = asyncio.new_event_loop()
                   try:
                       response = loop.run_until_complete(
                           llm_interface.generate_completion(
                               prompt=prompt,
                               model=self.wits_model,
                               stop=stop
                           )
                       )
                       return response
                   finally:
                       loop.close()

               async def _acall(self, prompt: str, stop: Optional[List[str]] = None) -> str:
                   """Run the LLM on the given prompt and input asynchronously."""
                   # Create LLM interface
                   llm_interface = LLMInterface(self.wits_config)

                   # Run the LLM
                   response = ""
                   async for chunk in llm_interface.generate_completion(
                       prompt=prompt,
                       model=self.wits_model,
                       stop=stop
                   ):
                       response += chunk

                   return response

           # Create and return the wrapper
           return WitsV3LLM(wits_config=self.config, wits_model=model)
   ```

2. **Support Langchain Tools**
   ```python
   # In tools/langchain_tool.py
   from typing import Dict, List, Any, Optional
   import logging

   from core.base_tool import BaseTool
   from core.schemas import ToolResult
   from core.config import WitsV3Config
   from core.langchain_bridge import LangchainBridge

   class LangchainToolRegistry(BaseTool):
       """Tool for registering and using Langchain tools."""

       def __init__(self, config: WitsV3Config):
           super().__init__(config)
           self.name = "langchain_tool_registry"
           self.description = "Register and use Langchain tools"
           self.bridge = LangchainBridge(config)
           self.logger = logging.getLogger(__name__)
           self.registered_tools = {}

       def get_schema(self) -> Dict[str, Any]:
           return {
               "type": "object",
               "properties": {
                   "action": {
                       "type": "string",
                       "enum": ["register", "list", "execute"],
                       "description": "Action to perform"
                   },
                   "tool_name": {
                       "type": "string",
                       "description": "Name of the tool to register or execute"
                   },
                   "tool_description": {
                       "type": "string",
                       "description": "Description of the tool to register"
                   },
                   "tool_function": {
                       "type": "string",
                       "description": "Python function code for the tool"
                   },
                   "input": {
                       "type": "string",
                       "description": "Input for the tool execution"
                   }
               },
               "required": ["action"]
           }

       async def execute(self, **kwargs) -> ToolResult:
           action = kwargs.get("action")

           if action == "register":
               return await self._register_tool(kwargs)
           elif action == "list":
               return await self._list_tools()
           elif action == "execute":
               return await self._execute_tool(kwargs)
           else:
               return ToolResult(
                   success=False,
                   error=f"Unknown action: {action}"
               )

       async def _register_tool(self, kwargs: Dict[str, Any]) -> ToolResult:
           """Register a new Langchain tool."""
           tool_name = kwargs.get("tool_name")
           tool_description = kwargs.get("tool_description")
           tool_function = kwargs.get("tool_function")

           if not tool_name or not tool_description or not tool_function:
               return ToolResult(
                   success=False,
                   error="Missing required parameters: tool_name, tool_description, tool_function"
               )

           try:
               # Create a function from the provided code
               # This is potentially dangerous and should be properly sandboxed in production
               tool_code = f"""
               def {tool_name}_func(input_str):
                   {tool_function}
               """

               # Create a local namespace to execute the code
               local_namespace = {}
               exec(tool_code, {}, local_namespace)

               # Get the function from the namespace
               func = local_namespace.get(f"{tool_name}_func")
               if not func:
                   return ToolResult(
                       success=False,
                       error=f"Failed to create function: {tool_name}_func"
                   )

               # Create a Langchain tool
               from langchain.agents import Tool as LangchainTool
               lc_tool = LangchainTool(
                   name=tool_name,
                   description=tool_description,
                   func=func
               )

               # Convert to WitsV3 tool and register
               wits_tool = self.bridge.langchain_tool_to_wits_tool(lc_tool)
               self.registered_tools[tool_name] = wits_tool

               return ToolResult(
                   success=True,
                   result=f"Tool {tool_name} registered successfully"
               )
           except Exception as e:
               return ToolResult(
                   success=False,
                   error=f"Error registering tool: {str(e)}"
               )

       async def _list_tools(self) -> ToolResult:
           """List all registered Langchain tools."""
           tools = []
           for name, tool in self.registered_tools.items():
               tools.append({
                   "name": name,
                   "description": tool.description
               })

           return ToolResult(
               success=True,
               result=tools
           )

       async def _execute_tool(self, kwargs: Dict[str, Any]) -> ToolResult:
           """Execute a registered Langchain tool."""
           tool_name = kwargs.get("tool_name")
           input_str = kwargs.get("input", "")

           if not tool_name:
               return ToolResult(
                   success=False,
                   error="Missing required parameter: tool_name"
               )

           if tool_name not in self.registered_tools:
               return ToolResult(
                   success=False,
                   error=f"Tool not found: {tool_name}"
               )

           # Execute the tool
           tool = self.registered_tools[tool_name]
           return await tool.execute(input=input_str)
   ```

3. **Add Langchain Agent Integration**
   ```python
   # In core/langchain_agent.py
   from typing import Dict, List, Any, Optional
   import logging
   import asyncio

   from langchain.agents import AgentExecutor, initialize_agent, AgentType
   from langchain.memory import ConversationBufferMemory

   from core.config import WitsV3Config
   from core.langchain_bridge import LangchainBridge
   from core.base_tool import BaseTool
   from core.schemas import ToolResult

   class LangchainAgentTool(BaseTool):
       """Tool for running Langchain agents."""

       def __init__(self, config: WitsV3Config):
           super().__init__(config)
           self.name = "langchain_agent"
           self.description = "Run Langchain agents with various tools"
           self.bridge = LangchainBridge(config)
           self.logger = logging.getLogger(__name__)
           self.agents = {}

       def get_schema(self) -> Dict[str, Any]:
           return {
               "type": "object",
               "properties": {
                   "action": {
                       "type": "string",
                       "enum": ["create", "run", "list"],
                       "description": "Action to perform"
                   },
                   "agent_name": {
                       "type": "string",
                       "description": "Name of the agent to create or run"
                   },
                   "agent_type": {
                       "type": "string",
                       "enum": ["zero-shot-react-description", "conversational-react-description"],
                       "description": "Type of agent to create"
                   },
                   "tools": {
                       "type": "array",
                       "items": {
                           "type": "string"
                       },
                       "description": "List of tool names to use with the agent"
                   },
                   "input": {
                       "type": "string",
                       "description": "Input for the agent"
                   },
                   "model": {
                       "type": "string",
                       "description": "Optional model name to use"
                   }
               },
               "required": ["action"]
           }

       async def execute(self, **kwargs) -> ToolResult:
           action = kwargs.get("action")

           if action == "create":
               return await self._create_agent(kwargs)
           elif action == "run":
               return await self._run_agent(kwargs)
           elif action == "list":
               return await self._list_agents()
           else:
               return ToolResult(
                   success=False,
                   error=f"Unknown action: {action}"
               )

       async def _create_agent(self, kwargs: Dict[str, Any]) -> ToolResult:
           """Create a new Langchain agent."""
           agent_name = kwargs.get("agent_name")
           agent_type = kwargs.get("agent_type")
           tool_names = kwargs.get("tools", [])
           model = kwargs.get("model")

           if not agent_name or not agent_type:
               return ToolResult(
                   success=False,
                   error="Missing required parameters: agent_name, agent_type"
               )

           try:
               # Get the agent type
               if agent_type == "zero-shot-react-description":
                   agent_type_enum = AgentType.ZERO_SHOT_REACT_DESCRIPTION
               elif agent_type == "conversational-react-description":
                   agent_type_enum = AgentType.CONVERSATIONAL_REACT_DESCRIPTION
               else:
                   return ToolResult(
                       success=False,
                       error=f"Unsupported agent type: {agent_type}"
                   )

               # Get the tools
               from core.tool_registry import ToolRegistry
               tool_registry = ToolRegistry(self.config)
               wits_tools = []

               for tool_name in tool_names:
                   tool = tool_registry.get_tool(tool_name)
                   if tool:
                       wits_tools.append(tool)
                   else:
                       return ToolResult(
                           success=False,
                           error=f"Tool not found: {tool_name}"
                       )

               # Convert WitsV3 tools to Langchain tools
               lc_tools = [self.bridge.wits_tool_to_langchain_tool(tool) for tool in wits_tools]

               # Create the LLM
               llm = self.bridge.create_langchain_llm_wrapper(model)

               # Create memory for conversational agents
               memory = None
               if agent_type == "conversational-react-description":
                   memory = ConversationBufferMemory(memory_key="chat_history")

               # Initialize the agent
               agent = initialize_agent(
                   tools=lc_tools,
                   llm=llm,
                   agent=agent_type_enum,
                   memory=memory,
                   verbose=True
               )

               # Store the agent
               self.agents[agent_name] = agent

               return ToolResult(
                   success=True,
                   result=f"Agent {agent_name} created successfully with {len(lc_tools)} tools"
               )
           except Exception as e:
               return ToolResult(
                   success=False,
                   error=f"Error creating agent: {str(e)}"
               )

       async def _run_agent(self, kwargs: Dict[str, Any]) -> ToolResult:
           """Run a Langchain agent."""
           agent_name = kwargs.get("agent_name")
           input_str = kwargs.get("input", "")

           if not agent_name:
               return ToolResult(
                   success=False,
                   error="Missing required parameter: agent_name"
               )

           if agent_name not in self.agents:
               return ToolResult(
                   success=False,
                   error=f"Agent not found: {agent_name}"
               )

           try:
               # Get the agent
               agent = self.agents[agent_name]

               # Run the agent
               result = await self.bridge.run_langchain_chain(agent, {"input": input_str})

               return ToolResult(
                   success=True,
                   result=result.get("output", "No output")
               )
           except Exception as e:
               return ToolResult(
                   success=False,
                   error=f"Error running agent: {str(e)}"
               )

       async def _list_agents(self) -> ToolResult:
           """List all registered Langchain agents."""
           agents = []
           for name in self.agents:
               agents.append(name)

           return ToolResult(
               success=True,
               result=agents
           )
   ```

### 3. Background Agent Monitoring

**Description**: Implement a monitoring system for background agents to track performance and resource usage.

**Implementation Steps**:

1. **Create Agent Monitoring System**
   ```python
   # In core/agent_monitor.py
   from typing import Dict, List, Any, Optional
   import logging
   import time
   import asyncio
   import psutil
   import os
   from datetime import datetime

   from core.config import WitsV3Config

   class AgentMonitor:
       """Monitor background agents for performance and resource usage."""

       def __init__(self, config: WitsV3Config):
           self.config = config
           self.logger = logging.getLogger(__name__)
           self.agents = {}
           self.metrics = {}
           self.running = False
           self.monitor_task = None

       async def register_agent(self, agent_id: str, agent_type: str) -> None:
           """Register an agent for monitoring."""
           self.agents[agent_id] = {
               "agent_id": agent_id,
               "agent_type": agent_type,
               "start_time": datetime.now().isoformat(),
               "status": "registered"
           }

           self.metrics[agent_id] = {
               "cpu_usage": [],
               "memory_usage": [],
               "request_count": 0,
               "error_count": 0,
               "average_response_time": 0,
               "total_response_time": 0
           }

           self.logger.info(f"Agent {agent_id} registered for monitoring")

       async def start_agent(self, agent_id: str, process_id: Optional[int] = None) -> None:
           """Start monitoring an agent."""
           if agent_id not in self.agents:
               self.logger.error(f"Agent {agent_id} not registered")
               return

           self.agents[agent_id]["status"] = "running"
           self.agents[agent_id]["process_id"] = process_id
           self.agents[agent_id]["start_time"] = datetime.now().isoformat()

           self.logger.info(f"Agent {agent_id} started")

           # Start the monitoring task if not already running
           if not self.running:
               await self.start_monitoring()

       async def stop_agent(self, agent_id: str) -> None:
           """Stop monitoring an agent."""
           if agent_id not in self.agents:
               self.logger.error(f"Agent {agent_id} not registered")
               return

           self.agents[agent_id]["status"] = "stopped"
           self.agents[agent_id]["stop_time"] = datetime.now().isoformat()

           self.logger.info(f"Agent {agent_id} stopped")

       async def record_request(self, agent_id: str, response_time: float, error: bool = False) -> None:
           """Record a request to an agent."""
           if agent_id not in self.metrics:
               self.logger.error(f"Agent {agent_id} not registered")
               return

           metrics = self.metrics[agent_id]
           metrics["request_count"] += 1

           if error:
               metrics["error_count"] += 1

           # Update average response time
           metrics["total_response_time"] += response_time
           metrics["average_response_time"] = metrics["total_response_time"] / metrics["request_count"]

       async def start_monitoring(self) -> None:
           """Start the monitoring task."""
           if self.running:
               return

           self.running = True
           self.monitor_task = asyncio.create_task(self._monitor_loop())

       async def stop_monitoring(self) -> None:
           """Stop the monitoring task."""
           if not self.running:
               return

           self.running = False
           if self.monitor_task:
               self.monitor_task.cancel()
               try:
                   await self.monitor_task
               except asyncio.CancelledError:
                   pass

           self.monitor_task = None

       async def _monitor_loop(self) -> None:
           """Monitor loop to collect metrics."""
           while self.running:
               try:
                   await self._collect_metrics()
               except Exception as e:
                   self.logger.error(f"Error collecting metrics: {str(e)}")

               # Sleep for the monitoring interval
               await asyncio.sleep(self.config.monitoring.interval)

       async def _collect_metrics(self) -> None:
           """Collect metrics for all running agents."""
           for agent_id, agent in self.agents.items():
               if agent["status"] != "running":
                   continue

               # Get process metrics if process ID is available
               process_id = agent.get("process_id")
               if process_id:
                   try:
                       process = psutil.Process(process_id)

                       # Get CPU and memory usage
                       cpu_percent = process.cpu_percent(interval=0.1)
                       memory_info = process.memory_info()
                       memory_mb = memory_info.rss / (1024 * 1024)  # Convert to MB

                       # Store metrics
                       self.metrics[agent_id]["cpu_usage"].append({
                           "timestamp": datetime.now().isoformat(),
                           "value": cpu_percent
                       })

                       self.metrics[agent_id]["memory_usage"].append({
                           "timestamp": datetime.now().isoformat(),
                           "value": memory_mb
                       })

                       # Limit the number of stored metrics
                       max_metrics = self.config.monitoring.max_metrics
                       if len(self.metrics[agent_id]["cpu_usage"]) > max_metrics:
                           self.metrics[agent_id]["cpu_usage"] = self.metrics[agent_id]["cpu_usage"][-max_metrics:]

                       if len(self.metrics[agent_id]["memory_usage"]) > max_metrics:
                           self.metrics[agent_id]["memory_usage"] = self.metrics[agent_id]["memory_usage"][-max_metrics:]
                   except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                       self.logger.warning(f"Could not access process {process_id} for agent {agent_id}")

       async def get_agent_status(self, agent_id: str) -> Dict[str, Any]:
           """Get the status of an agent."""
           if agent_id not in self.agents:
               return {"error": f"Agent {agent_id} not registered"}

           agent = self.agents[agent_id]
           metrics = self.metrics[agent_id]

           # Calculate current metrics
           current_cpu = metrics["cpu_usage"][-1]["value"] if metrics["cpu_usage"] else 0
           current_memory = metrics["memory_usage"][-1]["value"] if metrics["memory_usage"] else 0

           # Calculate average metrics
           avg_cpu = sum(m["value"] for m in metrics["cpu_usage"]) / len(metrics["cpu_usage"]) if metrics["cpu_usage"] else 0
           avg_memory = sum(m["value"] for m in metrics["memory_usage"]) / len(metrics["memory_usage"]) if metrics["memory_usage"] else 0

           return {
               "agent_id": agent_id,
               "agent_type": agent["agent_type"],
               "status": agent["status"],
               "start_time": agent["start_time"],
               "stop_time": agent.get("stop_time"),
               "uptime": self._calculate_uptime(agent),
               "metrics": {
                   "current": {
                       "cpu_percent": current_cpu,
                       "memory_mb": current_memory
                   },
                   "average": {
                       "cpu_percent": avg_cpu,
                       "memory_mb": avg_memory,
                       "response_time_ms": metrics["average_response_time"] * 1000
                   },
                   "totals": {
                       "request_count": metrics["request_count"],
                       "error_count": metrics["error_count"],
                       "error_rate": metrics["error_count"] / metrics["request_count"] if metrics["request_count"] > 0 else 0
                   }
               }
           }

       async def get_all_agent_statuses(self) -> List[Dict[str, Any]]:
           """Get the status of all agents."""
           statuses = []
           for agent_id in self.agents:
               status = await self.get_agent_status(agent_id)
               statuses.append(status)

           return statuses

       def _calculate_uptime(self, agent: Dict[str, Any]) -> float:
           """Calculate the uptime of an agent in seconds."""
           start_time = datetime.fromisoformat(agent["start_time"])

           if agent["status"] == "stopped" and "stop_time" in agent:
               end_time = datetime.fromisoformat(agent["stop_time"])
           else:
               end_time = datetime.now()

           return (end_time - start_time).total_seconds()
   ```

2. **Create Monitoring Dashboard**
   ```python
   # In gui/api/monitoring.py
   from fastapi import APIRouter, Depends, HTTPException
   from typing import List, Dict, Any

   from core.config import WitsV3Config
   from core.agent_monitor import AgentMonitor

   # Create router
   router = APIRouter(prefix="/monitoring", tags=["monitoring"])

   # Initialize components
   config = WitsV3Config()
   agent_monitor = AgentMonitor(config)

   @router.get("/agents")
   async def get_all_agents() -> Dict[str, Any]:
       """Get status of all agents."""
       statuses = await agent_monitor.get_all_agent_statuses()
       return {"agents": statuses}

   @router.get("/agents/{agent_id}")
   async def get_agent(agent_id: str) -> Dict[str, Any]:
       """Get status of a specific agent."""
       status = await agent_monitor.get_agent_status(agent_id)
       if "error" in status:
           raise HTTPException(status_code=404, detail=status["error"])

       return status

   @router.post("/agents/{agent_id}/start")
   async def start_agent(agent_id: str, process_id: int = None) -> Dict[str, Any]:
       """Start monitoring an agent."""
       await agent_monitor.start_agent(agent_id, process_id)
       return {"message": f"Agent {agent_id} started"}

   @router.post("/agents/{agent_id}/stop")
   async def stop_agent(agent_id: str) -> Dict[str, Any]:
       """Stop monitoring an agent."""
       await agent_monitor.stop_agent(agent_id)
       return {"message": f"Agent {agent_id} stopped"}

   @router.post("/agents/{agent_id}/record")
   async def record_request(agent_id: str, response_time: float, error: bool = False) -> Dict[str, Any]:
       """Record a request to an agent."""
       await agent_monitor.record_request(agent_id, response_time, error)
       return {"message": f"Request recorded for agent {agent_id}"}
   ```

3. **Add Background Agent Manager**
   ```python
   # In agents/background_agent_manager.py
   from typing import Dict, List, Any, Optional
   import logging
   import asyncio
   import os
   import signal
   import subprocess
   from datetime import datetime

   from core.config import WitsV3Config
   from core.agent_monitor import AgentMonitor

   class BackgroundAgentManager:
       """Manage background agents."""

       def __init__(self, config: WitsV3Config):
           self.config = config
           self.logger = logging.getLogger(__name__)
           self.monitor = AgentMonitor(config)
           self.agents = {}

       async def start_agent(self, agent_type: str, agent_config: Dict[str, Any]) -> Dict[str, Any]:
           """Start a background agent."""
           # Generate a unique agent ID
           agent_id = f"{agent_type}-{datetime.now().strftime('%Y%m%d%H%M%S')}"

           # Register the agent with the monitor
           await self.monitor.register_agent(agent_id, agent_type)

           # Create the agent process
           try:
               # Create the command to run the agent
               cmd = [
                   "python",
                   "run_background_agent.py",
                   "--agent-id", agent_id,
                   "--agent-type", agent_type
               ]

               # Add any additional configuration
               for key, value in agent_config.items():
                   cmd.extend([f"--{key}", str(value)])

               # Start the process
               process = subprocess.Popen(
                   cmd,
                   stdout=subprocess.PIPE,
                   stderr=subprocess.PIPE,
                   text=True
               )

               # Store the agent information
               self.agents[agent_id] = {
                   "agent_id": agent_id,
                   "agent_type": agent_type,
                   "process": process,
                   "config": agent_config,
                   "start_time": datetime.now().isoformat(),
                   "status": "running"
               }

               # Start monitoring the agent
               await self.monitor.start_agent(agent_id, process.pid)

               self.logger.info(f"Started background agent {agent_id} (PID: {process.pid})")

               return {
                   "agent_id": agent_id,
                   "agent_type": agent_type,
                   "process_id": process.pid,
                   "status": "running"
               }
           except Exception as e:
               self.logger.error(f"Error starting background agent: {str(e)}")
               return {
                   "error": f"Failed to start agent: {str(e)}"
               }

       async def stop_agent(self, agent_id: str) -> Dict[str, Any]:
           """Stop a background agent."""
           if agent_id not in self.agents:
               return {"error": f"Agent {agent_id} not found"}

           agent = self.agents[agent_id]
           process = agent["process"]

           try:
               # Send termination signal to the process
               process.terminate()

               # Wait for the process to terminate
               try:
                   process.wait(timeout=5)
               except subprocess.TimeoutExpired:
                   # Force kill if it doesn't terminate
                   process.kill()

               # Update agent status
               agent["status"] = "stopped"
               agent["stop_time"] = datetime.now().isoformat()

               # Stop monitoring the agent
               await self.monitor.stop_agent(agent_id)

               self.logger.info(f"Stopped background agent {agent_id}")

               return {
                   "agent_id": agent_id,
                   "status": "stopped"
               }
           except Exception as e:
               self.logger.error(f"Error stopping background agent: {str(e)}")
               return {
                   "error": f"Failed to stop agent: {str(e)}"
               }

       async def get_agent_status(self, agent_id: str) -> Dict[str, Any]:
           """Get the status of a background agent."""
           if agent_id not in self.agents:
               return {"error": f"Agent {agent_id} not found"}

           # Get the agent information
           agent = self.agents[agent_id]

           # Get the monitoring status
           monitor_status = await self.monitor.get_agent_status(agent_id)

           # Check if the process is still running
           process = agent["process"]
           is_running = process.poll() is None

           if is_running and agent["status"] != "running":
               agent["status"] = "running"
           elif not is_running and agent["status"] == "running":
               agent["status"] = "stopped"
               agent["stop_time"] = datetime.now().isoformat()

           return {
               "agent_id": agent_id,
               "agent_type": agent["agent_type"],
               "process_id": process.pid,
               "status": agent["status"],
               "start_time": agent["start_time"],
               "stop_time": agent.get("stop_time"),
               "config": agent["config"],
               "monitoring": monitor_status.get("metrics", {})
           }

       async def get_all_agent_statuses(self) -> List[Dict[str, Any]]:
           """Get the status of all background agents."""
           statuses = []
           for agent_id in self.agents:
               status = await self.get_agent_status(agent_id)
               statuses.append(status)

           return statuses
   ```

## Implementation Schedule

| Task | Start Date | End Date | Owner |
|------|------------|----------|-------|
| Web UI Prototype | June 26, 2025 | July 2, 2025 | TBD |
| Langchain Integration | June 28, 2025 | July 5, 2025 | TBD |
| Background Agent Monitoring | July 3, 2025 | July 9, 2025 | TBD |

## Success Criteria

- Web UI provides a functional interface for interacting with WitsV3
- Langchain tools and agents can be used within WitsV3
- Background agents can be monitored for performance and resource usage
- All components are integrated with the existing WitsV3 architecture
- All tests pass with the new implementations

## Dependencies

- Existing WitsV3 components
- FastAPI and React for Web UI
- Langchain library for integration
- Psutil for process monitoring

## Risks and Mitigations

| Risk | Mitigation |
|------|------------|
| Web UI security vulnerabilities | Implement proper authentication and authorization |
| Langchain compatibility issues | Test with specific Langchain versions and document requirements |
| Resource-intensive background agents | Implement resource limits and monitoring alerts |
| Integration complexity | Create comprehensive tests and documentation |

## Next Steps

After completing Phase 4, the team will:

1. Update documentation with the new features
2. Run comprehensive tests to verify stability
3. Begin work on Phase 5: Performance Optimization
