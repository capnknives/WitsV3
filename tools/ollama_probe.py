import aiohttp
import asyncio
from typing import List, Dict, Optional, Any
import json
import sys
from datetime import datetime

async def probe_endpoint(session: aiohttp.ClientSession, url: str, method: str = "GET", json_data: Optional[Dict[str, Any]] = None) -> Dict:
    """Probe a single endpoint and return its status."""
    try:
        async with session.request(method, url, json=json_data) as response:
            return {
                "url": url,
                "method": method,
                "status": response.status,
                "headers": dict(response.headers),
                "content": await response.text() if response.status < 400 else None,
                "timestamp": datetime.now().isoformat()
            }
    except Exception as e:
        return {
            "url": url,
            "method": method,
            "status": "error",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }

async def probe_ollama_endpoints(base_url: str = "http://localhost:11434") -> List[Dict]:
    """Probe all known Ollama endpoints."""
    endpoints = [
        # Health check endpoints
        ("/api/health", "GET"),
        ("/api/version", "GET"),

        # Model management endpoints
        ("/api/tags", "GET"),

        # Core functionality endpoints
        ("/api/generate", "POST", {"model": "llama3", "prompt": "test"}),
        ("/api/chat", "POST", {"model": "llama3", "messages": [{"role": "user", "content": "test"}]}),
        ("/api/embeddings", "POST", {"model": "nomic-embed-text", "prompt": "test"}),

        # Alternative health check endpoints
        ("/health", "GET"),
        ("/healthz", "GET"),
        ("/", "GET")
    ]

    async with aiohttp.ClientSession() as session:
        tasks = []
        for endpoint in endpoints:
            if len(endpoint) == 2:
                url, method = endpoint
                tasks.append(probe_endpoint(session, f"{base_url}{url}", method))
            else:
                url, method, json_data = endpoint
                tasks.append(probe_endpoint(session, f"{base_url}{url}", method, json_data))

        results = await asyncio.gather(*tasks)
        return results

def print_results(results: List[Dict]):
    """Print the probe results in a formatted way."""
    print("\nOllama Server Probe Results")
    print("=" * 80)

    # Group results by status
    successful = []
    failed = []
    errors = []

    for result in results:
        if result["status"] == "error":
            errors.append(result)
        elif isinstance(result["status"], int) and result["status"] < 400:
            successful.append(result)
        else:
            failed.append(result)

    # Print successful endpoints
    if successful:
        print("\n✅ Successful Endpoints:")
        print("-" * 40)
        for result in successful:
            print(f"\nEndpoint: {result['url']}")
            print(f"Method: {result['method']}")
            print(f"Status: {result['status']}")
            if result.get('content'):
                try:
                    content = json.loads(result['content'])
                    print("Content:", json.dumps(content, indent=2))
                except:
                    print("Content:", result['content'])

    # Print failed endpoints
    if failed:
        print("\n❌ Failed Endpoints:")
        print("-" * 40)
        for result in failed:
            print(f"\nEndpoint: {result['url']}")
            print(f"Method: {result['method']}")
            print(f"Status: {result['status']}")
            if result.get('content'):
                print("Error Response:", result['content'])

    # Print error endpoints
    if errors:
        print("\n⚠️ Error Endpoints:")
        print("-" * 40)
        for result in errors:
            print(f"\nEndpoint: {result['url']}")
            print(f"Method: {result['method']}")
            print(f"Error: {result['error']}")

    print("\n" + "=" * 80)

async def main():
    """Main function to run the probe."""
    print("Probing Ollama server endpoints...")
    try:
        results = await probe_ollama_endpoints()
        print_results(results)
    except Exception as e:
        print(f"Error during probe: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())
