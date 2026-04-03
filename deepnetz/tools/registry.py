"""
Tool Registry — manages available tools and routes tool calls.
"""

import json
from typing import Dict, List, Optional
from deepnetz.tools.base import Tool, ToolResult
from deepnetz.tools.search import WebSearchTool


class ToolRegistry:
    """Central registry for all available tools."""

    def __init__(self):
        self._tools: Dict[str, Tool] = {}
        # Register built-in tools
        self.register(WebSearchTool())

    def register(self, tool: Tool):
        """Register a tool."""
        self._tools[tool.name] = tool

    def get(self, name: str) -> Optional[Tool]:
        """Get a tool by name."""
        return self._tools.get(name)

    def list_tools(self) -> List[Tool]:
        """List all registered tools."""
        return list(self._tools.values())

    def to_openai_tools(self) -> List[Dict]:
        """Convert all tools to OpenAI function calling format."""
        return [t.to_openai_schema() for t in self._tools.values()]

    def execute(self, name: str, arguments: Dict) -> ToolResult:
        """Execute a tool by name with given arguments."""
        tool = self._tools.get(name)
        if not tool:
            return ToolResult(content="", success=False,
                            error=f"Unknown tool: {name}")
        try:
            return tool.execute(**arguments)
        except Exception as e:
            return ToolResult(content="", success=False, error=str(e))

    def parse_tool_calls(self, text: str) -> List[Dict]:
        """
        Parse tool calls from model output.
        Supports multiple formats:
        - Qwen: <tool_call>{"name": ..., "arguments": ...}</tool_call>
        - Llama: {"name": ..., "parameters": ...}
        - Generic: ```json\n{"tool": ..., "args": ...}\n```
        """
        calls = []

        # Qwen format: <tool_call>...</tool_call>
        import re
        for match in re.finditer(r'<tool_call>(.*?)</tool_call>', text, re.DOTALL):
            try:
                data = json.loads(match.group(1).strip())
                calls.append({
                    "name": data.get("name", ""),
                    "arguments": data.get("arguments", data.get("parameters", {}))
                })
            except json.JSONDecodeError:
                pass

        # JSON block format
        for match in re.finditer(r'\{[^{}]*"(?:name|tool|function)"[^{}]*\}', text):
            try:
                data = json.loads(match.group(0))
                name = data.get("name") or data.get("tool") or data.get("function", "")
                args = data.get("arguments") or data.get("parameters") or data.get("args", {})
                if name and name not in [c["name"] for c in calls]:
                    calls.append({"name": name, "arguments": args})
            except json.JSONDecodeError:
                pass

        return calls
