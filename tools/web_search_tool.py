"""
Web Search Tool for WitsV3.
Provides web search capabilities using DuckDuckGo.
"""

import logging
from typing import Any, Dict, List, Optional
import aiohttp
from urllib.parse import quote_plus

from core.base_tool import BaseTool
from core.schemas import ToolCall

logger = logging.getLogger(__name__)

class WebSearchTool(BaseTool):
    """Tool for performing web searches using DuckDuckGo."""
    
    def __init__(self):
        """Initialize the web search tool."""
        super().__init__(
            name="web_search",
            description="Search the web using DuckDuckGo. Returns relevant search results."
        )
        self.base_url = "https://api.duckduckgo.com/"
        
    async def execute(self, query: str, max_results: int = 5) -> List[Dict[str, Any]]:
        """
        Execute a web search.
        
        Args:
            query: The search query
            max_results: Maximum number of results to return
            
        Returns:
            List of search results with title, link, and snippet
        """
        try:
            encoded_query = quote_plus(query)
            params = {
                "q": encoded_query,
                "format": "json",
                "no_html": 1,
                "skip_disambig": 1
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.get(self.base_url, params=params) as response:
                    if response.status != 200:
                        logger.error(f"Web search failed with status {response.status}")
                        return []
                        
                    data = await response.json()
                    
                    results = []
                    if "RelatedTopics" in data:
                        for topic in data["RelatedTopics"]:
                            if "Text" in topic and "FirstURL" in topic:
                                results.append({
                                    "title": topic.get("Text", ""),
                                    "link": topic.get("FirstURL", ""),
                                    "snippet": topic.get("Text", "")
                                })
                                
                                if len(results) >= max_results:
                                    break
                    
                    return results
                    
        except Exception as e:
            logger.error(f"Error performing web search: {e}")
            return []
            
    def get_schema(self) -> Dict[str, Any]:
        """Get the tool's schema for LLM consumption."""
        return {
            "name": "web_search",
            "description": "Search the web using DuckDuckGo",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query"
                    },
                    "max_results": {
                        "type": "integer",
                        "description": "Maximum number of results to return",
                        "default": 5
                    }
                },
                "required": ["query"]
            }
        } 