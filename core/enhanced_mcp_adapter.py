# core/enhanced_mcp_adapter.py
"""
Enhanced MCP Adapter with dynamic tool discovery and management
Supports automatic tool ecosystem building and composition
"""

import asyncio
import json
import logging
import os
import subprocess
import tempfile
from typing import Dict, Any, List, Optional, Set
from pathlib import Path
from dataclasses import dataclass, field
import aiohttp
import yaml

from .mcp_adapter import MCPAdapter, MCPServer, MCPTool, StdioMCPClient
from .schemas import ToolCall, ToolResult

logger = logging.getLogger(__name__)


@dataclass
class ToolCapability:
    """Represents a capability that tools can provide"""
    name: str
    description: str
    required_inputs: List[str]
    expected_outputs: List[str]
    tags: Set[str]


@dataclass
class MCPToolRegistry:
    """Enhanced MCP tool registry with capabilities and metadata"""
    tools: Dict[str, MCPTool] = field(default_factory=dict)
    capabilities: Dict[str, List[str]] = field(default_factory=dict)  # capability -> tool names
    tool_metadata: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    tool_dependencies: Dict[str, List[str]] = field(default_factory=dict)


class EnhancedMCPAdapter(MCPAdapter):
    """
    Enhanced MCP adapter with dynamic discovery, composition, and ecosystem management
    """
    
    def __init__(self, config_path: Optional[str] = None):
        super().__init__()
        self.config_path = config_path or "mcp_ecosystem.yaml"
        self.tool_registry = MCPToolRegistry()
        self.available_servers = {}
        self.tool_workflows = {}
        self.ecosystem_config = {}
        
    async def initialize_ecosystem(self):
        """Initialize the MCP ecosystem with automatic discovery"""
        await self.load_ecosystem_config()
        await self.discover_available_tools()
        await self.auto_configure_servers()
        await self.build_capability_map()
        logger.info("MCP ecosystem initialized")
    
    async def load_ecosystem_config(self):
        """Load ecosystem configuration"""
        if os.path.exists(self.config_path):
            with open(self.config_path, 'r') as f:
                self.ecosystem_config = yaml.safe_load(f) or {}
        else:
            # Create default config
            self.ecosystem_config = {
                'auto_discovery': True,
                'server_configs': {},
                'tool_preferences': {},
                'security_settings': {
                    'sandbox_mode': True,
                    'allowed_domains': ['localhost'],
                    'resource_limits': {
                        'max_memory': '512MB',
                        'max_cpu_time': 30
                    }
                }
            }
            await self.save_ecosystem_config()
    
    async def save_ecosystem_config(self):
        """Save ecosystem configuration"""
        with open(self.config_path, 'w') as f:
            yaml.dump(self.ecosystem_config, f, default_flow_style=False)
    
    async def discover_available_tools(self):
        """Discover available MCP tools and servers"""
        discovered_tools = []
        
        # Check npm packages for MCP servers
        npm_servers = await self._discover_npm_mcp_servers()
        discovered_tools.extend(npm_servers)
        
        # Check GitHub for MCP tools
        github_servers = await self._discover_github_mcp_tools()
        discovered_tools.extend(github_servers)
        
        # Check local directories
        local_servers = await self._discover_local_mcp_tools()
        discovered_tools.extend(local_servers)
        
        # Update available servers
        for server_info in discovered_tools:
            self.available_servers[server_info['name']] = server_info
        
        logger.info(f"Discovered {len(discovered_tools)} MCP tools/servers")
    
    async def _discover_npm_mcp_servers(self) -> List[Dict[str, Any]]:
        """Discover MCP servers available via npm"""
        servers = []
        
        try:
            # Search npm for MCP packages
            result = await asyncio.create_subprocess_exec(
                'npm', 'search', '@modelcontextprotocol', '--json',
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            stdout, stderr = await result.communicate()
            
            if result.returncode == 0:
                packages = json.loads(stdout.decode())
                for package in packages:
                    servers.append({
                        'name': package['name'],
                        'type': 'npm',
                        'version': package['version'],
                        'description': package.get('description', ''),
                        'install_command': ['npx', '-y', package['name']],
                        'source': 'npm'
                    })
        except Exception as e:
            logger.warning(f"Failed to discover npm MCP servers: {e}")
        
        return servers
    
    async def _discover_github_mcp_tools(self) -> List[Dict[str, Any]]:
        """Discover MCP tools from GitHub repositories"""
        servers = []
        
        try:
            # Search GitHub API for MCP repositories
            async with aiohttp.ClientSession() as session:
                search_url = "https://api.github.com/search/repositories"
                params = {
                    'q': 'model-context-protocol OR mcp-server',
                    'sort': 'stars',
                    'order': 'desc'
                }
                
                async with session.get(search_url, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        
                        for repo in data.get('items', [])[:20]:  # Top 20 repos
                            servers.append({
                                'name': repo['name'],
                                'type': 'github',
                                'description': repo.get('description', ''),
                                'url': repo['html_url'],
                                'clone_url': repo['clone_url'],
                                'stars': repo['stargazers_count'],
                                'source': 'github'
                            })
        except Exception as e:
            logger.warning(f"Failed to discover GitHub MCP tools: {e}")
        
        return servers
    
    async def _discover_local_mcp_tools(self) -> List[Dict[str, Any]]:
        """Discover local MCP tools and servers"""
        servers = []
        
        # Check common locations for MCP tools
        search_paths = [
            Path.home() / '.mcp',
            Path('./mcp_tools'),
            Path('./tools/mcp'),
        ]
        
        for search_path in search_paths:
            if search_path.exists():
                for item in search_path.iterdir():
                    if item.is_dir() and (item / 'mcp.json').exists():
                        try:
                            with open(item / 'mcp.json') as f:
                                config = json.load(f)
                            
                            servers.append({
                                'name': config.get('name', item.name),
                                'type': 'local',
                                'path': str(item),
                                'description': config.get('description', ''),
                                'command': config.get('command', []),
                                'source': 'local'
                            })
                        except Exception as e:
                            logger.debug(f"Failed to read MCP config from {item}: {e}")
        
        return servers
    
    async def auto_configure_servers(self):
        """Automatically configure and connect to discovered servers"""
        if not self.ecosystem_config.get('auto_discovery', True):
            return
        
        # Prioritize servers based on configuration
        priority_order = [
            'filesystem', 'web-search', 'database', 'git', 'docker',
            'calendar', 'email', 'slack', 'notion'
        ]
        
        configured_count = 0
        for priority_name in priority_order:
            for server_name, server_info in self.available_servers.items():
                if priority_name.lower() in server_name.lower() and configured_count < 10:
                    success = await self._configure_and_connect_server(server_info)
                    if success:
                        configured_count += 1
                        break
        
        logger.info(f"Auto-configured {configured_count} MCP servers")
    
    async def _configure_and_connect_server(self, server_info: Dict[str, Any]) -> bool:
        """Configure and connect to a specific server"""
        try:
            server_name = server_info['name']
            
            # Skip if already connected
            if server_name in self.clients:
                return True
            
            # Create server configuration
            if server_info['type'] == 'npm':
                server_config = MCPServer(
                    name=server_name,
                    command=server_info['install_command']
                )
            
            elif server_info['type'] == 'local':
                command = server_info.get('command', ['python', str(Path(server_info['path']) / 'main.py')])
                server_config = MCPServer(
                    name=server_name,
                    command=command,
                    working_directory=server_info['path']
                )
            
            elif server_info['type'] == 'github':
                # Clone and setup GitHub repo
                clone_path = await self._clone_github_repo(server_info)
                if not clone_path:
                    return False
                
                server_config = MCPServer(
                    name=server_name,
                    command=['node', 'index.js'],  # Default assumption
                    working_directory=str(clone_path)
                )
            
            else:
                logger.warning(f"Unknown server type: {server_info['type']}")
                return False
            
            # Connect to server
            success = await self.add_server(server_config)
            if success:
                # Store metadata
                self.tool_registry.tool_metadata[server_name] = server_info
                
                # Update ecosystem config
                self.ecosystem_config['server_configs'][server_name] = {
                    'enabled': True,
                    'auto_configured': True,
                    'source': server_info['source']
                }
                await self.save_ecosystem_config()
            
            return success
            
        except Exception as e:
            logger.error(f"Failed to configure server {server_info['name']}: {e}")
            return False
    
    async def _clone_github_repo(self, server_info: Dict[str, Any]) -> Optional[Path]:
        """Clone a GitHub repository for MCP server"""
        try:
            # Create temp directory for cloning
            clone_dir = Path(tempfile.gettempdir()) / 'mcp_repos' / server_info['name']
            clone_dir.mkdir(parents=True, exist_ok=True)
            
            # Clone repository
            result = await asyncio.create_subprocess_exec(
                'git', 'clone', server_info['clone_url'], str(clone_dir),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            await result.communicate()
            
            if result.returncode == 0:
                # Try to install dependencies
                if (clone_dir / 'package.json').exists():
                    await asyncio.create_subprocess_exec(
                        'npm', 'install',
                        cwd=str(clone_dir)
                    )
                elif (clone_dir / 'requirements.txt').exists():
                    await asyncio.create_subprocess_exec(
                        'pip', 'install', '-r', 'requirements.txt',
                        cwd=str(clone_dir)
                    )
                
                return clone_dir
            
        except Exception as e:
            logger.error(f"Failed to clone GitHub repo {server_info['name']}: {e}")
        
        return None
    
    async def build_capability_map(self):
        """Build a map of capabilities to tools"""
        self.tool_registry.capabilities.clear()
        
        for tool_name, tool in self.tool_registry.tools.items():
            # Extract capabilities from tool description and schema
            capabilities = self._extract_tool_capabilities(tool)
            
            for capability in capabilities:
                if capability not in self.tool_registry.capabilities:
                    self.tool_registry.capabilities[capability] = []
                self.tool_registry.capabilities[capability].append(tool_name)
        
        logger.info(f"Built capability map with {len(self.tool_registry.capabilities)} capabilities")
    
    def _extract_tool_capabilities(self, tool: MCPTool) -> List[str]:
        """Extract capabilities from a tool's description and schema"""
        capabilities = []
        
        # Common capability patterns
        capability_patterns = {
            'file': ['read', 'write', 'create', 'delete', 'list'],
            'web': ['search', 'fetch', 'browse', 'scrape'],
            'data': ['analyze', 'transform', 'validate', 'export'],
            'communication': ['send', 'receive', 'notify', 'message'],
            'system': ['execute', 'monitor', 'control', 'configure'],
            'ai': ['generate', 'analyze', 'classify', 'translate'],
            'database': ['query', 'insert', 'update', 'backup'],
            'api': ['call', 'authenticate', 'webhook', 'rest']
        }
        
        tool_text = (tool.description + " " + str(tool.input_schema)).lower()
        
        for category, actions in capability_patterns.items():
            for action in actions:
                if action in tool_text or category in tool_text:
                    capabilities.append(f"{category}_{action}")
        
        # Add generic capabilities based on tool name
        if 'file' in tool.name.lower():
            capabilities.extend(['file_read', 'file_write'])
        if 'web' in tool.name.lower():
            capabilities.extend(['web_search', 'web_fetch'])
        if 'db' in tool.name.lower() or 'database' in tool.name.lower():
            capabilities.extend(['database_query'])
        
        return list(set(capabilities))
    
    async def find_tools_for_capability(self, capability: str) -> List[MCPTool]:
        """Find tools that provide a specific capability"""
        tool_names = self.tool_registry.capabilities.get(capability, [])
        return [self.tool_registry.tools[name] for name in tool_names 
                if name in self.tool_registry.tools]
    
    async def compose_workflow(self, workflow_name: str, steps: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Compose a workflow from multiple tools"""
        workflow = {
            'name': workflow_name,
            'steps': steps,
            'created_at': asyncio.get_event_loop().time()
        }
        
        # Validate workflow steps
        for i, step in enumerate(steps):
            tool_name = step.get('tool')
            if tool_name not in self.tool_registry.tools:
                return {'error': f"Tool {tool_name} not found in step {i}"}
            
            # Check if previous step output matches current step input
            if i > 0:
                prev_outputs = steps[i-1].get('expected_outputs', [])
                curr_inputs = step.get('inputs', {})
                # Add validation logic here
        
        self.tool_workflows[workflow_name] = workflow
        return {'success': True, 'workflow': workflow}
    
    async def execute_workflow(self, workflow_name: str, initial_inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a composed workflow"""
        if workflow_name not in self.tool_workflows:
            return {'error': f"Workflow {workflow_name} not found"}
        
        workflow = self.tool_workflows[workflow_name]
        results = []
        current_data = initial_inputs.copy()
        
        for i, step in enumerate(workflow['steps']):
            try:
                tool_name = step['tool']
                tool_args = step.get('arguments', {})
                
                # Substitute data from previous steps
                processed_args = self._substitute_workflow_data(tool_args, current_data)
                
                # Execute tool
                tool_call = ToolCall(
                    call_id=f"workflow_{workflow_name}_step_{i}",
                    tool_name=tool_name,
                    arguments=processed_args
                )
                
                result = await self.call_tool(tool_call)
                results.append(result)
                
                # Update current data with result
                if result.success:
                    current_data.update({f"step_{i}_result": result.result})
                else:
                    return {
                        'error': f"Workflow failed at step {i}: {result.error}",
                        'partial_results': results
                    }
                    
            except Exception as e:
                return {
                    'error': f"Workflow execution failed at step {i}: {str(e)}",
                    'partial_results': results
                }
        
        return {
            'success': True,
            'results': results,
            'final_data': current_data
        }
    
    def _substitute_workflow_data(self, args: Dict[str, Any], data: Dict[str, Any]) -> Dict[str, Any]:
        """Substitute workflow data into tool arguments"""
        def substitute_value(value):
            if isinstance(value, str) and value.startswith('${') and value.endswith('}'):
                key = value[2:-1]
                return data.get(key, value)
            elif isinstance(value, dict):
                return {k: substitute_value(v) for k, v in value.items()}
            elif isinstance(value, list):
                return [substitute_value(item) for item in value]
            return value
        
        return {k: substitute_value(v) for k, v in args.items()}
    
    async def install_tool(self, tool_identifier: str) -> bool:
        """Install a tool by identifier (npm package, GitHub repo, etc.)"""
        try:
            if tool_identifier.startswith('https://github.com/'):
                # GitHub repository
                server_info = {
                    'name': tool_identifier.split('/')[-1],
                    'type': 'github',
                    'clone_url': tool_identifier,
                    'source': 'manual_install'
                }
                return await self._configure_and_connect_server(server_info)
            
            elif tool_identifier.startswith('@'):
                # npm package
                server_info = {
                    'name': tool_identifier,
                    'type': 'npm',
                    'install_command': ['npx', '-y', tool_identifier],
                    'source': 'manual_install'
                }
                return await self._configure_and_connect_server(server_info)
            
            else:
                # Try to find in available servers
                for server_name, server_info in self.available_servers.items():
                    if tool_identifier.lower() in server_name.lower():
                        return await self._configure_and_connect_server(server_info)
                
                logger.error(f"Tool identifier not recognized: {tool_identifier}")
                return False
                
        except Exception as e:
            logger.error(f"Failed to install tool {tool_identifier}: {e}")
            return False
    
    async def get_ecosystem_status(self) -> Dict[str, Any]:
        """Get the current status of the MCP ecosystem"""
        return {
            'connected_servers': len(self.clients),
            'available_tools': len(self.tool_registry.tools),
            'capabilities': len(self.tool_registry.capabilities),
            'workflows': len(self.tool_workflows),
            'discovered_servers': len(self.available_servers),
            'server_list': list(self.clients.keys()),
            'capability_list': list(self.tool_registry.capabilities.keys()),
            'workflow_list': list(self.tool_workflows.keys())
        }
    
    async def recommend_tools(self, task_description: str) -> List[Dict[str, Any]]:
        """Recommend tools for a given task description"""
        recommendations = []
        
        # Simple keyword matching for now - could be enhanced with embeddings
        task_lower = task_description.lower()
        
        # Score tools based on relevance to task
        for tool_name, tool in self.tool_registry.tools.items():
            score = 0
            tool_text = (tool.description + " " + tool.name).lower()
            
            # Keyword matching
            for word in task_lower.split():
                if word in tool_text:
                    score += 1
            
            # Capability matching
            for capability in self.tool_registry.capabilities:
                if any(word in capability for word in task_lower.split()):
                    if tool_name in self.tool_registry.capabilities[capability]:
                        score += 2
            
            if score > 0:
                recommendations.append({
                    'tool': tool,
                    'score': score,
                    'reason': f"Matches {score} keywords/capabilities"
                })
        
        # Sort by score
        recommendations.sort(key=lambda x: x['score'], reverse=True)
        return recommendations[:10]


# Test function
async def test_enhanced_mcp_adapter():
    """Test the enhanced MCP adapter functionality"""
    adapter = EnhancedMCPAdapter()
    
    try:
        # Initialize ecosystem
        await adapter.initialize_ecosystem()
        
        # Get status
        status = await adapter.get_ecosystem_status()
        print(f"Ecosystem status: {status}")
        
        # Test tool recommendations
        recommendations = await adapter.recommend_tools("I need to read files and search the web")
        print(f"Tool recommendations: {len(recommendations)}")
        
        # Test workflow composition
        workflow_result = await adapter.compose_workflow(
            "file_analysis",
            [
                {"tool": "read_file", "arguments": {"file_path": "${input_file}"}},
                {"tool": "analyze_text", "arguments": {"text": "${step_0_result}"}}
            ]
        )
        print(f"Workflow composition: {workflow_result}")
        
    except Exception as e:
        print(f"Test completed with expected errors: {e}")


if __name__ == "__main__":
    asyncio.run(test_enhanced_mcp_adapter())