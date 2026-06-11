# Copyright 2025-2026 Strands RL Contributors
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

"""Reward function for Terminal-Bench environment."""

from __future__ import annotations

import logging
import re
from typing import TYPE_CHECKING

from harbor.models.trial.paths import EnvironmentPaths

from strands_env.core.types import Action, RewardFunction, RewardResult, StepResult

if TYPE_CHECKING:
    from .env import TerminalBenchEnv

logger = logging.getLogger(__name__)

# Last absolute WORKDIR wins. Strips optional surrounding quotes.
# SETA's bundled `test.sh` asserts `$PWD != "/"` and exits 1 otherwise. e2b's
# `exec()` defaults to cwd=`/`, so without this we'd fail the guard before
# writing `reward.txt`. tb2 task images also tend to set WORKDIR (typically
# `/app`); resolving it here is a no-op-equivalent for those.
_WORKDIR_RE = re.compile(r'^\s*WORKDIR\s+["\']?(/[^"\'\s]*)["\']?\s*$', re.MULTILINE | re.IGNORECASE)


class TerminalBenchReward(RewardFunction):
    """Execute test scripts in Docker and compute binary reward (0 or 1)."""

    def __init__(self, env: TerminalBenchEnv) -> None:
        """Initialize a `TerminalBenchReward` instance."""
        self._env = env

    async def compute(self, action: Action, step_result: StepResult) -> RewardResult:
        """Run verification tests in Docker and return a binary reward."""
        try:
            reward = await self._run_verification()
            return RewardResult(reward=reward, info={"status": "success"})
        except Exception as e:
            logger.exception("Verification failed due to %s: %s", type(e).__name__, str(e))
            return RewardResult(reward=0.0, info={"status": "error", "message": str(e)})

    def _resolve_workdir(self) -> str | None:
        """Read the last absolute `WORKDIR` directive from the task's Dockerfile.

        Returns `None` if no Dockerfile exists or no absolute `WORKDIR` is set —
        in which case the caller should not pass an explicit cwd, letting the
        backend's default apply.
        """
        dockerfile = self._env.task_paths.environment_dir / "Dockerfile"
        if not dockerfile.exists():
            return None
        matches = _WORKDIR_RE.findall(dockerfile.read_text())
        return matches[-1] if matches else None

    async def _run_verification(self) -> float:
        """Upload tests, execute `test.sh` from the image's `WORKDIR`, parse reward."""
        assert self._env.docker_env is not None, "Docker environment not initialized"
        docker_env = self._env.docker_env
        task_paths = self._env.task_paths
        trial_paths = self._env.trial_paths
        timeout = self._env.timeout
        cwd = self._resolve_workdir()

        # Ensure verifier output dir exists in the sandbox. Some task images (e.g.
        # SETA's `FROM ubuntu:24.04`) don't pre-create `/logs/verifier`, so the
        # `tee` redirect and `test.sh`'s `echo > /logs/verifier/reward.txt` would
        # fail silently. tb2's alexgshaw images already create it, so this is a
        # no-op there.
        await docker_env.exec(
            f"mkdir -p {EnvironmentPaths.verifier_dir} {EnvironmentPaths.agent_dir}",
            timeout_sec=30,
        )
        await docker_env.upload_dir(source_dir=task_paths.tests_dir, target_dir="/tests")
        test_cmd = (
            'export PATH="$HOME/.local/bin:$PATH" && '
            f"bash /tests/test.sh 2>&1 | tee {EnvironmentPaths.verifier_dir}/test-stdout.txt"
        )
        # Pass cwd only when we have a resolved WORKDIR — otherwise backend default
        # applies (preserves prior tb2 behavior for tasks without a Dockerfile).
        exec_kwargs: dict = {"timeout_sec": timeout}
        if cwd is not None:
            exec_kwargs["cwd"] = cwd
        await docker_env.exec(test_cmd, **exec_kwargs)

        # Download results if not using mounted volumes
        if not docker_env.is_mounted:
            await docker_env.download_dir(
                source_dir=str(EnvironmentPaths.verifier_dir),
                target_dir=trial_paths.verifier_dir,
            )

        # Parse reward (1.0 if reward.txt contains value >= 1, else 0.0)
        reward_path = trial_paths.reward_text_path
        if reward_path.exists() and reward_path.stat().st_size > 0:
            return 1.0 if float(reward_path.read_text().strip()) >= 1.0 else 0.0
        logger.warning("No reward file at %s (cwd=%s)", reward_path, cwd)
        return 0.0
