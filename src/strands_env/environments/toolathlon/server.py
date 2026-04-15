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

"""Toolathlon workspace setup and preprocessing utilities."""

from __future__ import annotations

import logging
import os
import shutil
import socket
import subprocess
import sys
from pathlib import Path

from toolathlon_gym.utils.data_structures.task_config import TaskConfig
from toolathlon_gym.utils.general.helper import copy_folder_contents

logger = logging.getLogger(__name__)

GYM_ROOT = Path(os.environ.get("TOOLATHLON_GYM_ROOT", os.getcwd()))

# MCP servers that need pre-created workspace subdirectories.
_MCP_WORKSPACE_DIRS: list[tuple[str, str]] = [
    ("arxiv_local", "arxiv_local_storage"),
    ("memory", "memory"),
    ("playwright_with_chunk", ".playwright_output"),
]


async def setup_workspace(task_config: TaskConfig) -> str:
    """Create workspace directory and copy initial files.

    Mirrors ``TaskAgent._setup_workspace()`` using the GYM's own
    ``copy_folder_contents`` utility.

    Returns:
        Absolute path to the agent workspace.
    """
    workspace = os.path.abspath(task_config.agent_workspace)
    os.makedirs(workspace, exist_ok=True)

    init = task_config.initialization
    if init and init.workspace and os.path.exists(str(init.workspace)):
        await copy_folder_contents(str(init.workspace), workspace)

    for server_name, subdir in _MCP_WORKSPACE_DIRS:
        if server_name in (task_config.needed_mcp_servers or []):
            os.makedirs(os.path.join(workspace, subdir), exist_ok=True)

    return workspace


def _get_available_port() -> int:
    """Return a random available TCP port on localhost."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def run_preprocess(
    task_config: TaskConfig, *, db_name: str = "toolathlon_gym", temp_dir: Path | None = None
) -> int:
    """Run the task's preprocessing script with per-rollout isolation.

    When ``temp_dir`` is provided the task directory is copied there so
    each rollout's mock-file extraction and ``serve_static`` are fully
    independent.  A random port is assigned via the ``MOCK_PORT`` env var
    (preprocess scripts read it with ``os.environ.get("MOCK_PORT", default)``).

    Returns:
        The mock server port assigned to this rollout.
    """
    init = task_config.initialization
    mock_port = _get_available_port()
    if not (init and init.process_command):
        return mock_port

    task_dir_name = getattr(task_config, "task_dir", "") or ""
    task_root = GYM_ROOT / "tasks" / "finalpool" / task_dir_name

    # Copy task dir into temp_dir for per-rollout isolation.
    cwd = GYM_ROOT
    cmd = init.process_command.replace("python3 ", f"{sys.executable} ", 1)
    if temp_dir is not None and task_root.is_dir():
        rollout_task = temp_dir / "task"
        shutil.copytree(task_root, rollout_task)
        # Rewrite module path: tasks.finalpool.<task>.preprocess.main → task.preprocess.main
        cmd = cmd.replace(f"tasks.finalpool.{task_dir_name}", "task")
        cwd = temp_dir  # so `python -m task.preprocess.main` resolves

    cmd += f" --agent_workspace {task_config.agent_workspace}"
    launch_time = " ".join((task_config.launch_time or "").split()[:2])
    cmd += f' --launch_time "{launch_time}"'

    env = os.environ.copy()
    env["PGDATABASE"] = db_name
    env["MOCK_PORT"] = str(mock_port)

    result = subprocess.run(  # noqa: S602
        cmd, shell=True, capture_output=True, text=True, cwd=cwd, env=env,
    )
    if result.returncode != 0:
        logger.warning("Preprocess failed (rc=%d): %s", result.returncode, (result.stderr or "")[:300])

    return mock_port


OUTCOME_COMPLETED = "COMPLETED"
OUTCOME_AGENT_FAILED = "AGENT_FAILED"
OUTCOME_EVAL_ERROR = "EVAL_ERROR"


def run_evaluation(task_config: TaskConfig, *, db_name: str = "toolathlon_gym") -> tuple[float, dict]:
    """Execute the task's evaluation script.

    Args:
        task_config: The task configuration.
        db_name: Database name for the rollout (e.g. ``rollout_0``).  Injected
            as ``PGDATABASE`` so the evaluation script queries the correct database.

    Returns:
        A tuple of (reward, info). Info contains ``"outcome"`` — one of
        ``COMPLETED``, ``AGENT_FAILED``, or ``EVAL_ERROR`` (the evaluation
        script itself crashed).
    """
    cmd = task_config.evaluation.evaluation_command.replace("python3 ", f"{sys.executable} ", 1)
    cmd += f" --agent_workspace {task_config.agent_workspace}"
    if task_config.evaluation.groundtruth_workspace:
        cmd += f" --groundtruth_workspace {task_config.evaluation.groundtruth_workspace}"
    launch_time = " ".join((task_config.launch_time or "").split()[:2])
    cmd += f' --launch_time "{launch_time}"'

    env = os.environ.copy()
    env["PGDATABASE"] = db_name

    result = subprocess.run(  # noqa: S602
        cmd, shell=True, capture_output=True, text=True, cwd=GYM_ROOT, env=env,
    )
    if result.returncode == 0:
        return 1.0, {"outcome": OUTCOME_COMPLETED}

    logger.info("Evaluation failed (rc=%d): %s", result.returncode, (result.stderr or "")[:300])
    return 0.0, {"outcome": OUTCOME_AGENT_FAILED, "error": result.stderr or ""}
