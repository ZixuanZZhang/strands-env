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

"""Terminal-Bench environment using Harbor for container management and test execution."""

from __future__ import annotations

import uuid
from pathlib import Path
from typing import TYPE_CHECKING, Literal, TypeAlias

from harbor.environments.factory import EnvironmentFactory
from harbor.models.environment_type import EnvironmentType
from harbor.models.task.config import EnvironmentConfig as _HarborEnvironmentConfig
from harbor.models.task.paths import TaskPaths
from harbor.models.trial.paths import TrialPaths
from strands import tool
from typing_extensions import NotRequired, TypedDict, Unpack, override

from strands_env.core import Environment, ModelFactory
from strands_env.core.environment import EnvironmentConfig
from strands_env.core.types import RewardFunction

from .reward import TerminalBenchReward

if TYPE_CHECKING:
    from harbor.environments.base import BaseEnvironment

    HarborEnvironment: TypeAlias = BaseEnvironment

HarborEnvironmentConfig: TypeAlias = _HarborEnvironmentConfig


class TerminalBenchConfig(EnvironmentConfig):
    """Serializable configuration for `TerminalBenchEnv`.

    Backends:
        - "docker": Local Docker via `harbor`'s native `DockerEnvironment`.
        - "e2b": e2b sandbox via `harbor.environments.e2b.E2BEnvironment`.
            Endpoint + auth come from env vars `E2B_DOMAIN` / `E2B_API_KEY`
            (the e2b SDK reads these). Self-hosted clusters that haven't
            back-ported the upstream `/v3/templates` route must additionally
            supply `E2B_PREBAKED_TEMPLATES_PATH` pointing at a
            `{task_name: template_id}` JSON produced by an out-of-band bake.
    """

    task_id: str
    task_dir: str
    trial_dir: str
    timeout: NotRequired[int]
    backend: NotRequired[Literal["docker", "e2b"]]
    harbor_env_config: NotRequired[HarborEnvironmentConfig]
    e2b_backend_config: NotRequired[E2bBackendConfig]


class E2bBackendConfig(TypedDict, total=False):
    """Configuration for the e2b backend.

    Endpoint + auth (`E2B_DOMAIN`, `E2B_API_KEY`) come from process env vars;
    the e2b SDK reads them automatically. Fields below are e2b-specific
    config not in the standard env-var set.
    """

    # Pre-baked template id for the task. Optional: when omitted, the adapter
    # resolves the id from `templates_json` (or `E2B_PREBAKED_TEMPLATES_PATH`)
    # using the task name as the lookup key. Provide explicitly only when
    # overriding the bake mapping for a one-off.
    template_id: str
    # Path to templates.json produced by the bake script. Falls back to env var
    # `E2B_PREBAKED_TEMPLATES_PATH` when unset. Required (via either source)
    # unless `template_id` is provided directly.
    templates_json: str


class TerminalBenchEnv(Environment):
    """Terminal-Bench environment using Harbor for container management and test execution."""

    default_system_prompt_path = Path(__file__).parent / "system_prompt.md"

    def __init__(
        self,
        *,
        model_factory: ModelFactory,
        reward_fn: RewardFunction | None = None,
        **config: Unpack[TerminalBenchConfig],
    ):
        """Initialize a `TerminalBenchEnv` instance."""
        super().__init__(model_factory=model_factory, reward_fn=None, **config)  # type: ignore[misc]
        self.task_id: str = str(self.config["task_id"])
        self.task_paths = TaskPaths(Path(str(self.config["task_dir"])))
        self.trial_paths = TrialPaths(Path(str(self.config["trial_dir"])))
        self.timeout: int = int(self.config.get("timeout", 1200))
        self.backend: Literal["docker", "e2b"] = self.config.get("backend", "docker")
        self.harbor_env_config: HarborEnvironmentConfig = self.config.get(
            "harbor_env_config", HarborEnvironmentConfig()
        )
        self.e2b_backend_config: E2bBackendConfig = self.config.get("e2b_backend_config", {})
        self.docker_env: HarborEnvironment | None = None
        self.reward_fn = reward_fn or TerminalBenchReward(self)

    @override
    async def reset(self) -> None:
        """Build and start the container environment."""
        self.trial_paths.mkdir()
        session_id = f"{self.task_id}-{uuid.uuid4().hex[:8]}"

        force_build = True
        match self.backend:
            case "docker":
                self.docker_env = EnvironmentFactory.create_environment(
                    type=EnvironmentType.DOCKER,
                    environment_dir=self.task_paths.environment_dir,
                    environment_name=session_id,
                    session_id=session_id,
                    trial_paths=self.trial_paths,
                    task_env_config=self.harbor_env_config,
                )
            case "e2b":
                # Self-hosted e2b api forks lacking `/v3/templates` (which Harbor's
                # E2BEnvironment._create_template calls) need templates baked
                # out-of-band. PreBakedE2BEnvironment looks up the pre-baked id
                # and skips the auto-build path. See _e2b_pre_baked.py.
                from ._e2b_pre_baked import PreBakedE2BEnvironment, resolve_template_id

                template_id = self.e2b_backend_config.get("template_id") or resolve_template_id(
                    task_name=self.task_id,
                    template_map_path=self.e2b_backend_config.get("templates_json"),
                )
                # force_build is ignored by PreBakedE2BEnvironment; templates
                # are static once baked. Re-bake out-of-band to refresh.
                force_build = False
                self.docker_env = PreBakedE2BEnvironment(
                    environment_dir=self.task_paths.environment_dir,
                    environment_name=session_id,
                    session_id=session_id,
                    trial_paths=self.trial_paths,
                    task_env_config=self.harbor_env_config,
                    template_id=template_id,
                )

        await self.docker_env.start(force_build=force_build)

    @tool
    async def execute_command(self, command: str) -> str:
        """Execute a shell command in the environment.

        Args:
            command: The shell command to execute (e.g., "ls -la", "cat file.txt")

        Returns:
            Command output (stdout + stderr combined).
        """
        # TODO: Align the terminal command ouput with OpenHand's output format.
        if not self.docker_env:
            raise RuntimeError("Docker environment not initialized")
        result = await self.docker_env.exec(command, timeout_sec=self.timeout)
        output = result.stdout or ""
        if result.stderr:
            output += f"\n[stderr]: {result.stderr}"
        if result.return_code != 0:
            output += f"\n[exit code]: {result.return_code}"
        return output.strip() or "(no output)"

    @override
    def get_tools(self) -> list:
        """Return the execute_command tool."""
        return [self.execute_command]

    @override
    async def cleanup(self) -> None:
        """Stop and delete the Docker environment."""
        if self.docker_env:
            await self.docker_env.stop(delete=True)
            self.docker_env = None
