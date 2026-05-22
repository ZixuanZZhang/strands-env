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

"""Terminal-Bench environment driven by `EKSPodClient` directly — no Harbor dep.

Mirrors `strands_env.environments.terminal_bench.TerminalBenchEnv` (Strands
`Environment` + `execute_command` tool + binary reward via `bash test.sh`) but
uses our async `EKSPodClient` / `EKSPod` for container management instead of
Harbor's `DockerEnvironment` / `harbor-aws`'s `AWSEnvironment`.

Constraints:
- Image must already exist in a registry the cluster can pull (typically ECR);
  this env does NOT build Dockerfiles.
- Pod runs `sleep infinity`; all work happens via `kubectl exec`-style commands.
- `verifier_dir` lives in the pod under `/logs/verifier`; results are downloaded
  post-test.
"""

from __future__ import annotations

import logging
from pathlib import Path

from strands import tool
from typing_extensions import NotRequired, Unpack, override

from strands_env.core import Environment, ModelFactory
from strands_env.core.environment import EnvironmentConfig
from strands_env.core.types import RewardFunction
from strands_env.utils.eks import EKSPod, EKSPodClient

from .reward import TerminalBenchEKSReward

logger = logging.getLogger(__name__)


class TerminalBenchEKSConfig(EnvironmentConfig):
    """Serializable configuration for `TerminalBenchEKSEnv`."""

    task_id: str
    task_dir: str
    trial_dir: str
    image: str
    cluster_name: str
    timeout: NotRequired[int]
    verify_timeout: NotRequired[int]
    cpu: NotRequired[str]
    memory: NotRequired[str]
    storage: NotRequired[str]
    region: NotRequired[str]
    namespace: NotRequired[str]
    role_arn: NotRequired[str | None]
    ecr_pull_through_cache: NotRequired[str | None]


class TerminalBenchEKSEnv(Environment):
    """Terminal-Bench rollout env on EKS Fargate via `EKSPodClient`."""

    default_system_prompt_path = Path(__file__).parent / "system_prompt.md"

    def __init__(
        self,
        *,
        model_factory: ModelFactory,
        reward_fn: RewardFunction | None = None,
        client: EKSPodClient | None = None,
        **config: Unpack[TerminalBenchEKSConfig],
    ):
        """Initialize a `TerminalBenchEKSEnv` instance.

        Args:
            model_factory: Passthrough to `Environment`.
            reward_fn: Passthrough; defaults to `TerminalBenchEKSReward(self)`.
            client: Optional pre-built `EKSPodClient` shared across envs in the same
                process (saves token/session overhead). Caller owns its lifecycle when
                supplied; otherwise the env builds and closes its own client.
            **config: See `TerminalBenchEKSConfig`.
        """
        super().__init__(model_factory=model_factory, reward_fn=None, **config)  # type: ignore[misc]
        self.task_id: str = str(self.config["task_id"])
        self.task_dir: Path = Path(str(self.config["task_dir"]))
        self.trial_dir: Path = Path(str(self.config["trial_dir"]))
        self.image: str = str(self.config["image"])
        self.timeout: int = int(self.config.get("timeout", 1200))
        # `test.sh` often runs slower than individual agent commands, so callers
        # (e.g. the SETA eval driver) set this to ~2× `timeout`.
        self.verify_timeout: int = int(self.config.get("verify_timeout", self.timeout))
        self.cpu: str = str(self.config.get("cpu", "1"))
        self.memory: str = str(self.config.get("memory", "2Gi"))
        self.storage: str = str(self.config.get("storage", "50Gi"))

        self._owns_client = client is None
        self.client: EKSPodClient = client or EKSPodClient(
            cluster_name=str(self.config["cluster_name"]),
            region=str(self.config.get("region", "us-east-1")),
            namespace=str(self.config.get("namespace", "default")),
            role_arn=self.config.get("role_arn"),
            ecr_pull_through_cache=self.config.get("ecr_pull_through_cache"),
        )
        self.pod: EKSPod | None = None
        self.reward_fn = reward_fn or TerminalBenchEKSReward(self)

    @override
    async def reset(self) -> None:
        """Create the trial output dir and start a fresh pod for this episode."""
        self.trial_dir.mkdir(parents=True, exist_ok=True)
        # INFO (not DEBUG): one line per task at eval-driver level for tail -f progress.
        logger.info(
            "reset: task=%s image=%s cpu=%s memory=%s storage=%s",
            self.task_id,
            self.image,
            self.cpu,
            self.memory,
            self.storage,
        )
        self.pod = await self.client.start_pod(
            image=self.image,
            cpu=self.cpu,
            memory=self.memory,
            storage=self.storage,
            labels={"strands-env/task": self.task_id[:60]},
            startup_timeout_sec=600.0,
            name_prefix=f"tb-{self.task_id[:20]}",
        )
        logger.info("reset done: task=%s pod=%s", self.task_id, self.pod.name)

    @tool
    async def execute_command(self, command: str) -> str:
        """Execute a shell command in the environment.

        Args:
            command: The shell command to execute (e.g., "ls -la", "cat file.txt")

        Returns:
            Command output (stdout + stderr combined).
        """
        if self.pod is None:
            raise RuntimeError("Pod not started — call reset() first")
        result = await self.pod.exec(command, timeout_sec=self.timeout)
        output = result.stdout or ""
        if result.stderr:
            output += f"\n[stderr]: {result.stderr}"
        if result.return_code != 0:
            output += f"\n[exit code]: {result.return_code}"
        return output.strip() or "(no output)"

    @override
    def get_tools(self) -> list:
        """Return the `execute_command` tool."""
        return [self.execute_command]

    @override
    async def cleanup(self) -> None:
        """Stop the pod and (if owned) close the cluster client.

        Failures are logged but never raised: the eval driver's `evaluate_sample`
        would otherwise mark the whole sample as `aborted=True` even when the
        agent run + reward computation already succeeded, wasting work. Orphan
        pods are picked up by the cluster's pod-TTL janitor.
        """
        if self.pod is not None:
            pod_name = self.pod.name
            try:
                await self.pod.stop()
                logger.info("cleanup: task=%s pod=%s stopped", self.task_id, pod_name)
            except Exception as e:
                logger.warning(
                    "cleanup: task=%s pod=%s stop failed: %r — leaving pod for janitor",
                    self.task_id,
                    pod_name,
                    e,
                )
            self.pod = None
        if self._owns_client:
            try:
                await self.client.close()
            except Exception as e:
                logger.warning("cleanup: client.close failed: %r", e)
