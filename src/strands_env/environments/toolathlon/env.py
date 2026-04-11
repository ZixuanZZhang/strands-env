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

"""Toolathlon-GYM environment — MCP tools backed by local PostgreSQL mock services."""

from __future__ import annotations

import asyncio
import contextlib
import logging
import os
import re
import shutil
from pathlib import Path
from typing import Any

import psycopg2
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from toolathlon_gym.utils.data_structures.task_config import TaskConfig
from toolathlon_gym.utils.mcp.tool_servers import build_mcp_clients
from typing_extensions import NotRequired, Unpack, override

from strands_env.core.environment import Environment, EnvironmentConfig
from strands_env.core.models import ModelFactory
from strands_env.core.types import RewardFunction

from .reward import ToolathlonRewardFunction
from .server import GYM_ROOT, run_preprocess, setup_workspace
from .tools import ToolathlonMCPTool, build_local_tools

logger = logging.getLogger(__name__)

# Default database credentials (from toolathlon_gym docker-compose.yml).
_DB_HOST = "localhost"
_DB_PORT = 5432
_DB_USER = "eigent"
_DB_PASSWORD = "camel"
_DB_TEMPLATE = "toolathlon_gym"


class ToolathlonConfig(EnvironmentConfig):
    """Serializable configuration for `ToolathlonEnvironment`."""

    task_dir: str
    temp_dir: str
    tool_call_timeout: NotRequired[int]
    rollout_id: NotRequired[int]


class ToolathlonEnvironment(Environment):
    """MCP environment backed by Toolathlon-GYM's local PostgreSQL mock services.

    Each ``reset()`` creates an isolated database (``rollout_<id>``) from the
    template via ``CREATE DATABASE ... TEMPLATE``, enabling safe concurrent
    rollouts — each actor's MCP servers connect to their own database and
    filesystem workspace.

    Notes:
        - `reset()` resets the rollout database, builds a per-task workspace,
          runs preprocessing, starts MCP servers as stdio subprocesses, and
          discovers tools.
        - `cleanup()` closes all MCP sessions/transports and removes the temp
          directory.
    """

    def __init__(
        self,
        *,
        model_factory: ModelFactory,
        reward_fn: RewardFunction | None = None,
        **config: Unpack[ToolathlonConfig],
    ):
        """Initialize a `ToolathlonEnvironment` instance."""
        super().__init__(
            model_factory=model_factory,
            reward_fn=reward_fn or ToolathlonRewardFunction(env=self),
            **config,  # type: ignore[misc]
        )
        self.task_dir = str(self.config["task_dir"])
        self.temp_dir = Path(str(self.config["temp_dir"]))
        self.tool_call_timeout = int(self.config.get("tool_call_timeout", 60))
        self.rollout_id = int(self.config.get("rollout_id", 0))
        self.db_name = f"rollout_{self.rollout_id}"
        self.task_config: TaskConfig | None = None
        self._exit_stack: contextlib.AsyncExitStack | None = None
        self.mcp_tools: list[ToolathlonMCPTool] = []
        self.local_tools: list[Any] = []

    @override
    async def reset(self) -> None:
        """Reset rollout database, build workspace, start MCP servers, discover tools."""
        os.chdir(GYM_ROOT)

        # 1. Reset rollout database from template
        await asyncio.to_thread(self._reset_database)

        # 2. Build task config — dump directly into temp_dir
        self.task_config = TaskConfig.build(
            self.task_dir,
            agent_short_name="strands",
            single_turn_mode=True,
            global_task_config={"dump_path": str(self.temp_dir), "direct_to_dumps": True},
        )

        # 3. Setup workspace
        workspace = await setup_workspace(self.task_config)

        # 4. Run preprocess (seed data into rollout database)
        await asyncio.to_thread(run_preprocess, self.task_config, db_name=self.db_name)

        # 5. Start MCP servers and discover tools
        self.mcp_tools = await self._connect_mcp_servers()

        # 6. Build local tools
        self.local_tools = build_local_tools(self.task_config, workspace)

        # 7. Set system prompt from task config
        if self.task_config.system_prompts and self.task_config.system_prompts.agent:
            self.system_prompt = self.task_config.system_prompts.agent

    def _reset_database(self) -> None:
        """Drop and recreate the rollout database from the template."""
        conn = psycopg2.connect(
            host=_DB_HOST, port=_DB_PORT, dbname="postgres",
            user=_DB_USER, password=_DB_PASSWORD,
        )
        conn.autocommit = True
        try:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT pg_terminate_backend(pid) FROM pg_stat_activity "
                    "WHERE datname = %s AND pid <> pg_backend_pid()", (self.db_name,),
                )
                cur.execute(f"DROP DATABASE IF EXISTS {self.db_name}")  # noqa: S608
                cur.execute(f"CREATE DATABASE {self.db_name} TEMPLATE {_DB_TEMPLATE}")  # noqa: S608
            logger.info("Reset database %s from template %s", self.db_name, _DB_TEMPLATE)
        finally:
            conn.close()

    async def _connect_mcp_servers(self) -> list[ToolathlonMCPTool]:
        """Build MCP clients via GYM, extract configs, connect via ``mcp`` SDK."""
        assert self.task_config is not None  # noqa: S101
        camel_clients = build_mcp_clients(
            needed_servers=self.task_config.needed_mcp_servers,
            agent_workspace=self.task_config.agent_workspace,
            config_dir=str(GYM_ROOT / "configs" / "mcp_servers"),
            task_dir=str(GYM_ROOT / "tasks" / "finalpool" / self.task_dir),
        )

        db_env = {
            "PGHOST": _DB_HOST, "PG_HOST": _DB_HOST,
            "PGPORT": str(_DB_PORT), "PG_PORT": str(_DB_PORT),
            "PGDATABASE": self.db_name, "PG_DATABASE": self.db_name,
            "PGUSER": _DB_USER, "PG_USER": _DB_USER,
            "PGPASSWORD": _DB_PASSWORD, "PG_PASSWORD": _DB_PASSWORD,
        }
        stack = contextlib.AsyncExitStack()
        tools: list[ToolathlonMCPTool] = []
        try:
            for client in camel_clients:
                cfg = client.config
                env = {**(cfg.env or {}), **db_env}
                params = StdioServerParameters(
                    command=cfg.command,
                    args=cfg.args or [],
                    env=env,
                    cwd=getattr(cfg, "cwd", None),
                )
                transport = stdio_client(params)
                read_stream, write_stream = await stack.enter_async_context(transport)
                session = await stack.enter_async_context(ClientSession(read_stream, write_stream))
                init_result = await session.initialize()
                server_name = re.sub(r"[^a-zA-Z0-9_-]", "-", init_result.serverInfo.name)

                result = await session.list_tools()
                for mcp_tool in result.tools:
                    mcp_tool.name = f"{server_name}_{mcp_tool.name}"
                    tools.append(ToolathlonMCPTool(mcp_tool, session, timeout=self.tool_call_timeout))

                logger.info("Connected MCP server %s: %d tools", server_name, len(result.tools))
        except BaseException:
            await stack.aclose()
            raise

        self._exit_stack = stack
        logger.info("Total MCP tools discovered: %d", len(tools))
        return tools

    @override
    def get_tools(self) -> list:
        """Return MCP tools + local tools discovered during `reset()`."""
        return list(self.mcp_tools) + list(self.local_tools)

    @override
    async def cleanup(self) -> None:
        """Close all MCP sessions/transports and remove temp directory."""
        self.mcp_tools = []
        self.local_tools = []
        self.task_config = None
        if self._exit_stack:
            with contextlib.suppress(Exception):
                await self._exit_stack.aclose()
            self._exit_stack = None
        await asyncio.to_thread(shutil.rmtree, self.temp_dir, True)
