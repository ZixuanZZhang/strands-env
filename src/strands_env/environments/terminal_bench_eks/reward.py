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

"""Reward function for the EKS Terminal-Bench environment."""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING

from strands_env.core.types import Action, RewardFunction, RewardResult, StepResult

if TYPE_CHECKING:
    from .env import TerminalBenchEKSEnv

logger = logging.getLogger(__name__)

#: Path on the pod where verifier outputs (test stdout, reward.txt) accumulate.
#: Matches Harbor's `EnvironmentPaths.verifier_dir` so terminal-bench task
#: `test.sh` scripts that write `/logs/verifier/reward.txt` work unmodified.
_POD_VERIFIER_DIR = "/logs/verifier"


class TerminalBenchEKSReward(RewardFunction):
    """Upload `tests/`, run `bash test.sh` in the pod, download outputs, parse reward."""

    def __init__(self, env: TerminalBenchEKSEnv) -> None:
        """Initialize a `TerminalBenchEKSReward` instance."""
        self._env = env

    async def compute(self, action: Action, step_result: StepResult) -> RewardResult:
        """Run verification in the pod and return a binary reward."""
        task_id = self._env.task_id
        t_start = time.monotonic()
        try:
            reward = await self._run_verification()
            logger.info("verify: task=%s reward=%.1f dur=%.1fs", task_id, reward, time.monotonic() - t_start)
            return RewardResult(reward=reward, info={"status": "success"})
        except Exception as e:
            logger.exception("Verification failed due to %s: %s", type(e).__name__, str(e))
            return RewardResult(reward=0.0, info={"status": "error", "message": str(e)})

    async def _run_verification(self) -> float:
        """Upload tests, exec `test.sh`, download verifier dir, parse reward."""
        env = self._env
        if env.pod is None:
            raise RuntimeError("Pod not started — call reset() first")

        await env.pod.upload_dir(env.task_dir / "tests", "/tests")

        test_cmd = (
            f"mkdir -p {_POD_VERIFIER_DIR} && "
            'export PATH="$HOME/.local/bin:$PATH" && '
            f"bash /tests/test.sh 2>&1 | tee {_POD_VERIFIER_DIR}/test-stdout.txt"
        )
        await env.pod.exec(test_cmd, timeout_sec=env.verify_timeout)

        # Fargate has no host fs sharing — always download.
        local_verifier_dir = env.trial_dir / "verifier"
        await env.pod.download_dir(_POD_VERIFIER_DIR, local_verifier_dir)

        reward_path = local_verifier_dir / "reward.txt"
        if reward_path.exists() and reward_path.stat().st_size > 0:
            return 1.0 if float(reward_path.read_text().strip()) >= 1.0 else 0.0
        raise RuntimeError(f"verification produced no reward file at {reward_path}")
