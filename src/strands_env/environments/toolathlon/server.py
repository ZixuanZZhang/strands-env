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
import subprocess
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


def run_preprocess(task_config: TaskConfig, *, db_name: str = "toolathlon_gym") -> None:
    """Run the task's preprocessing script (database setup, seed data, etc.).

    Args:
        task_config: The task configuration.
        db_name: Database name for the rollout (e.g. ``rollout_0``).  Injected
            as ``PGDATABASE`` so the preprocess script seeds the correct database.
    """
    init = task_config.initialization
    if not (init and init.process_command):
        return

    cmd = init.process_command
    cmd += f" --agent_workspace {task_config.agent_workspace}"
    launch_time = " ".join((task_config.launch_time or "").split()[:2])
    cmd += f' --launch_time "{launch_time}"'

    env = os.environ.copy()
    env["PGDATABASE"] = db_name

    result = subprocess.run(  # noqa: S602
        cmd, shell=True, capture_output=True, text=True, cwd=GYM_ROOT, env=env,
    )
    if result.returncode != 0:
        logger.warning("Preprocess failed (rc=%d): %s", result.returncode, (result.stderr or "")[:300])


def run_evaluation(task_config: TaskConfig, *, db_name: str = "toolathlon_gym") -> float:
    """Execute the task's evaluation script.

    Args:
        task_config: The task configuration.
        db_name: Database name for the rollout (e.g. ``rollout_0``).  Injected
            as ``PGDATABASE`` so the evaluation script queries the correct database.

    Returns:
        ``1.0`` on pass, ``0.0`` on fail.
    """
    cmd = task_config.evaluation.evaluation_command
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
        return 1.0

    logger.info("Evaluation failed (rc=%d): %s", result.returncode, (result.stderr or "")[:300])
    return 0.0
