"""
Internet Search Tool — DuckDuckGo-based web search.

No API key needed. Rate-limited to be respectful.
"""

from typing import Dict, Any
from deepnetz.tools.base import Tool, ToolResult


class WebSearchTool(Tool):
    """Search the internet using DuckDuckGo."""

    @property
    def name(self) -> str:
        return "web_search"

    @property
    def description(self) -> str:
        return "Search the internet for current information. Use when the user asks about recent events, facts you're not sure about, or anything that needs up-to-date information."

    @property
    def parameters(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query"
                },
                "max_results": {
                    "type": "integer",
                    "description": "Maximum number of results (default 5)",
                    "default": 5
                }
            },
            "required": ["query"]
        }

    def execute(self, query: str = "", max_results: int = 5, **kwargs) -> ToolResult:
        if not query:
            return ToolResult(content="", success=False, error="No query provided")

        try:
            from duckduckgo_search import DDGS
        except ImportError:
            # Fallback: use urllib
            return self._fallback_search(query, max_results)

        try:
            with DDGS() as ddgs:
                results = list(ddgs.text(query, max_results=max_results))
                if not results:
                    return ToolResult(content="No results found.", data={"results": []})

                text_parts = []
                for i, r in enumerate(results, 1):
                    title = r.get("title", "")
                    body = r.get("body", "")
                    url = r.get("href", "")
                    text_parts.append(f"{i}. {title}\n   {body}\n   URL: {url}")

                return ToolResult(
                    content="\n\n".join(text_parts),
                    data={"results": results}
                )
        except Exception as e:
            return ToolResult(content="", success=False, error=str(e))

    def _fallback_search(self, query: str, max_results: int) -> ToolResult:
        """Fallback search without duckduckgo-search library."""
        import urllib.request
        import urllib.parse
        import json

        try:
            encoded = urllib.parse.quote(query)
            url = f"https://api.duckduckgo.com/?q={encoded}&format=json&no_redirect=1"
            req = urllib.request.Request(url, headers={"User-Agent": "DeepNetz/0.9"})
            resp = urllib.request.urlopen(req, timeout=10)
            data = json.loads(resp.read().decode())

            results = []
            # Abstract
            if data.get("Abstract"):
                results.append(f"1. {data['Heading']}\n   {data['Abstract']}\n   URL: {data.get('AbstractURL', '')}")

            # Related topics
            for i, topic in enumerate(data.get("RelatedTopics", [])[:max_results], len(results)+1):
                if isinstance(topic, dict) and "Text" in topic:
                    results.append(f"{i}. {topic['Text']}\n   URL: {topic.get('FirstURL', '')}")

            if results:
                return ToolResult(content="\n\n".join(results))
            return ToolResult(content=f"Search results for: {query} (install duckduckgo-search for better results)")

        except Exception as e:
            return ToolResult(content="", success=False, error=str(e))
