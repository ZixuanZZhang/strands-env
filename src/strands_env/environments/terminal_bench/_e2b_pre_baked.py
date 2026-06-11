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

"""Pre-baked e2b template adapter.

Some self-hosted e2b api forks (notably those based on
aws-samples/sample-e2b-on-aws @ e2b-dev/infra 0c35ed5, mid-2024) do not implement
the ``/v3/templates`` route that Harbor's ``E2BEnvironment._create_template``
calls. The route was added in upstream e2b after that fork point and is not
back-portable as a small patch.

Workaround: bake templates out-of-band using the older ``/templates`` route the
api server supports, record the resulting ``templateID`` per task in a
``templates.json`` file, and override ``_create_template`` to be a no-op +
``start`` to look up the pre-baked id.

When the self-hosted fork catches up to upstream's ``/v3/templates``, this
module can be removed and ``terminal_bench/env.py`` can switch back to the
factory's plain ``EnvironmentType.E2B`` path.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import TYPE_CHECKING

from harbor.environments.e2b import E2BEnvironment
from typing_extensions import override

if TYPE_CHECKING:
    from harbor.models.task.config import EnvironmentConfig as HarborEnvironmentConfig
    from harbor.models.trial.paths import TrialPaths


def _load_template_map(path: str | Path) -> dict[str, str]:
    """Load a {task_name: template_id} mapping from a JSON file.

    The file is produced by the out-of-band bake script after a successful
    bake of each task's template against the self-hosted e2b api.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(
            f"Pre-baked templates file not found at {p}. "
            "Bake templates against the e2b cluster first.",
        )
    return json.loads(p.read_text())


def resolve_template_id(task_name: str, template_map_path: str | Path | None = None) -> str:
    """Resolve a tb2/swebench task name to its pre-baked e2b template id.

    Args:
        task_name: The task's name (e.g. ``"fix-git"``). Matches the directory
            name in the tb2 dataset and the ``Task.name`` field after Harbor's
            mapper has run.
        template_map_path: Path to ``templates.json``. If None, reads from
            ``E2B_PREBAKED_TEMPLATES_PATH``.

    Raises:
        FileNotFoundError: If the templates file does not exist.
        KeyError: If the task name has no entry. The bake script must run for
            new tasks before they can be evaluated.
    """
    if template_map_path is None:
        template_map_path = os.environ.get("E2B_PREBAKED_TEMPLATES_PATH")
    if template_map_path is None:
        raise RuntimeError(
            "E2B_PREBAKED_TEMPLATES_PATH not set and no template_map_path provided. "
            "Set it to the templates.json produced by the bake script.",
        )
    mapping = _load_template_map(template_map_path)
    # Tolerate both Harbor's `<benchmark>/<task>` form (e.g. `terminal-bench/fix-git`)
    # and the bare directory name (`fix-git`). The bake script writes the bare
    # directory name; the evaluator passes Harbor's full Task.name.
    candidates = [task_name, task_name.split("/")[-1]]
    for candidate in candidates:
        if candidate in mapping:
            return mapping[candidate]
    raise KeyError(
        f"Task {task_name!r} has no pre-baked template. "
        f"Tried: {candidates}. "
        f"Known tasks: {sorted(mapping.keys())[:10]}{'...' if len(mapping) > 10 else ''}. "
        "Bake the missing task first.",
    )


class PreBakedE2BEnvironment(E2BEnvironment):
    """E2BEnvironment that skips Harbor's auto-build path.

    Construction is identical to ``E2BEnvironment`` except a ``template_id``
    kwarg pins the pre-baked template the sandbox should boot from.
    """

    # Harbor's reward.py checks ``docker_env.is_mounted`` to decide whether
    # results need to be downloaded out of the environment. Mounted
    # filesystems (Docker on Linux + bind mounts) skip the download because
    # the host already sees the files. e2b sandboxes are remote — files only
    # exist inside the microVM until downloaded, so this must be False.
    is_mounted = False

    def __init__(
        self,
        environment_dir: Path,
        environment_name: str,
        session_id: str,
        trial_paths: TrialPaths,
        task_env_config: HarborEnvironmentConfig,
        *args,
        template_id: str,
        **kwargs,
    ):
        super().__init__(
            environment_dir=environment_dir,
            environment_name=environment_name,
            session_id=session_id,
            trial_paths=trial_paths,
            task_env_config=task_env_config,
            *args,
            **kwargs,
        )
        self._pre_baked_template_id = template_id

    @override
    async def start(self, force_build: bool) -> None:
        # Skip _does_template_exist + _create_template. Use the pre-baked id.
        self._template_name = self._pre_baked_template_id
        if force_build:
            self.logger.warning(
                "force_build=True requested but PreBakedE2BEnvironment ignores it. "
                "Re-bake the template out-of-band if needed.",
            )
        await self._create_sandbox()
        if not self._sandbox:
            raise RuntimeError("Sandbox not found but was just created.")
        await self.ensure_dirs(self._mount_targets(writable_only=True))

    @override
    async def _create_template(self) -> None:  # type: ignore[override]
        # Surface a precise error if Harbor ever falls into this path (it
        # shouldn't, since we override start; but defensive).
        raise RuntimeError(
            "PreBakedE2BEnvironment does not auto-build templates. "
            "Templates must be pre-baked before eval.",
        )
