"""AWM environment — synthetic FastAPI + SQLite server exposed via MCP.

Each AWM task is a FastAPI app with SQLite state, wrapped with ``fastapi_mcp``
to expose REST endpoints as MCP tools.  The evaluator/trainer prepares
``AWMConfig`` fields (DB files, envs JSONL); this environment starts the server
and connects the MCPClient.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import os
import shutil
import signal
import sys
from dataclasses import dataclass
from pathlib import Path

from awm.tools import get_random_available_port
from mcp.client.streamable_http import streamable_http_client
from strands.tools.mcp import MCPClient

from strands_env.core.models import ModelFactory
from strands_env.core.types import RewardFunction
from strands_env.environments.mcp import MCPEnvironment

from .reward import AWMRewardFunction

logger = logging.getLogger(__name__)

AWM_BIN = str(Path(sys.executable).parent / "awm")
SERVER_STARTUP_TIMEOUT = 30


@dataclass
class AWMConfig:
    """Configuration for task-dependent arguments in ``AWMEnvironment``.

    Attributes:
        scenario: Scenario name.
        envs_path: Path to gen_envs.jsonl (contains scenario, db_path, full_code).
        work_db_path: Working DB copy the server writes to.
        initial_db_path: Read-only DB snapshot for reward verification.
        temp_dir: Temp directory for server artifacts.
    """

    scenario: str
    envs_path: str
    work_db_path: str
    initial_db_path: str
    temp_dir: str


class AWMEnvironment(MCPEnvironment):
    """MCP environment backed by an AWM FastAPI server subprocess.

    ``reset()`` starts the server and connects the MCPClient.
    ``cleanup()`` kills the server process group, stops the MCPClient, and removes the temp dir.
    """

    def __init__(
        self,
        *,
        model_factory: ModelFactory,
        config: AWMConfig,
        reward_fn: RewardFunction | None = None,
        max_tool_iters: int | None = None,
        max_tool_calls: int | None = None,
        verbose: bool = False,
    ):
        super().__init__(
            model_factory=model_factory,
            reward_fn=reward_fn or AWMRewardFunction(),
            max_tool_iters=max_tool_iters,
            max_tool_calls=max_tool_calls,
            verbose=verbose,
        )
        self.config = config
        self._server_proc: asyncio.subprocess.Process | None = None

    async def reset(self) -> None:
        """Start the AWM server, wait for readiness, and connect the MCPClient."""
        port = get_random_available_port()
        self._server_proc = await asyncio.create_subprocess_exec(
            AWM_BIN,
            "env",
            "start",
            "--scenario",
            self.config.scenario,
            "--envs_load_path",
            self.config.envs_path,
            "--db_path",
            self.config.work_db_path,
            "--port",
            str(port),
            "--temp_server_path",
            str(Path(self.config.temp_dir) / "server.py"),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            start_new_session=True,
        )
        await self._wait_for_server()
        logger.info("AWM server started for %s on port %d (pid %d)", self.config.scenario, port, self._server_proc.pid)

        url = f"http://localhost:{port}/mcp"
        self._mcp_client = MCPClient(lambda: streamable_http_client(url))
        await super().reset()

    async def _wait_for_server(self) -> None:
        """Wait for uvicorn to signal readiness via stderr, with timeout."""

        async def _read_until_ready():
            async for line in self._server_proc.stderr:
                if b"Application startup complete" in line:
                    return
            raise RuntimeError(f"AWM server for {self.config.scenario} exited before startup completed")

        await asyncio.wait_for(_read_until_ready(), timeout=SERVER_STARTUP_TIMEOUT)

    async def cleanup(self) -> None:
        """Kill the server process group, stop the MCPClient, and remove the temp dir.
        The server is killed first so that MCPClient.stop() doesn't hang
        waiting on a response from an unresponsive server.
        """
        self._kill_server()
        await super().cleanup()
        if self.config.temp_dir:
            shutil.rmtree(self.config.temp_dir, ignore_errors=True)

    def _kill_server(self) -> None:
        if self._server_proc and self._server_proc.returncode is None:
            with contextlib.suppress(ProcessLookupError):
                os.killpg(self._server_proc.pid, signal.SIGKILL)
            self._server_proc = None
