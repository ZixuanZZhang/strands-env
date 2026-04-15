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

"""Reward function for Toolathlon environment."""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING

from strands_env.core.types import Action, RewardFunction, RewardResult, StepResult

from .server import run_evaluation

if TYPE_CHECKING:
    from .env import ToolathlonEnvironment

logger = logging.getLogger(__name__)


class ToolathlonRewardFunction(RewardFunction):
    """Run the task's ``evaluation/main.py`` script for binary reward (0 or 1)."""

    def __init__(self, env: ToolathlonEnvironment | None = None) -> None:
        """Initialize a `ToolathlonRewardFunction` instance."""
        self._env = env

    async def compute(self, action: Action, step_result: StepResult) -> RewardResult:
        """Run evaluation script and return binary reward."""
        assert self._env is not None and self._env.task_config is not None  # noqa: S101
        try:
            reward, info = await asyncio.to_thread(
                run_evaluation, self._env.task_config, db_name=self._env.db_name,
            )
            return RewardResult(reward=reward, info=info)
        except Exception as e:
            logger.exception("Evaluation failed: %s", e)
            return RewardResult(
                reward=0.0,
                info={"outcome": "EVAL_ERROR", "error": str(e), "error_type": type(e).__name__},
            )
