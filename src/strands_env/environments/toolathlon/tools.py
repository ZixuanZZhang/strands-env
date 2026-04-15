# Copyright 2025-2026 Horizon RL Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Toolathlon MCP tool adapter and local tool builders."""

from __future__ import annotations

import json
from datetime import timedelta
from typing import Any, Literal

from mcp import ClientSession
from mcp.types import TextContent
from mcp.types import Tool as MCPToolDef
from strands import tool
from strands.types.tools import ToolResultContent
from toolathlon_gym.utils.aux_tools.basic import sleep as gym_sleep
from toolathlon_gym.utils.aux_tools.overlong_tool_manager import make_overlong_tools
from toolathlon_gym.utils.aux_tools.python_interpretor import make_python_execute
from toolathlon_gym.utils.data_structures.task_config import TaskConfig
from typing_extensions import override

from strands_env.tools.mcp_tool import MCPToolAdapter


class ToolathlonMCPTool(MCPToolAdapter):
    """MCP tool backed by a stdio `ClientSession`.

    The strands Agent sees the prefixed name (``{server}_{tool}``) for
    uniqueness across servers, but the MCP server only knows the original
    unprefixed name.  ``_original_name`` stores the latter so that
    ``call_tool`` dispatches correctly.
    """

    def __init__(
        self, mcp_tool: MCPToolDef, session: ClientSession, *, original_name: str, timeout: int = 60
    ):
        """Initialize a `ToolathlonMCPTool` instance."""
        super().__init__(mcp_tool, timeout=timedelta(seconds=timeout))
        self._session = session
        self._original_name = original_name

    @override
    async def call_tool(
        self, name: str, args: dict[str, Any]
    ) -> tuple[list[ToolResultContent], Literal["success", "error"]]:
        """Execute tool via MCP session with automatic type coercion."""
        properties = self._mcp_tool.inputSchema.get("properties", {})
        for key, value in args.items():
            if isinstance(value, str) and properties.get(key, {}).get("type") in (
                "array",
                "object",
                "integer",
                "number",
                "boolean",
            ):
                try:
                    args[key] = json.loads(value)
                except (json.JSONDecodeError, ValueError):
                    pass
        result = await self._session.call_tool(self._original_name, args, self._timeout)
        content = [ToolResultContent(text=item.text) for item in result.content if isinstance(item, TextContent)]
        status: Literal["success", "error"] = "error" if result.isError else "success"
        return content, status


def build_local_tools(task_config: TaskConfig, workspace: str) -> list:
    """Build local tools from GYM's factory functions.

    Args:
        task_config: The task configuration with ``needed_local_tools``.
        workspace: Absolute path to the agent workspace.

    Returns:
        List of strands tool-wrapped callables.
    """
    needed = set(task_config.needed_local_tools or [])
    tools: list = []

    if "python_execute" in needed:
        tools.append(tool(make_python_execute(workspace)))

    if "handle_overlong_tool_outputs" in needed:
        save_fn, view_fn = make_overlong_tools(workspace)
        tools.append(tool(save_fn))
        tools.append(tool(view_fn))

    if "sleep" in needed:
        tools.append(tool(gym_sleep))

    return tools
