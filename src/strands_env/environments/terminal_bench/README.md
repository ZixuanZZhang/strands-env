# Terminal-Bench Environment

An environment for [Terminal-Bench](https://github.com/terminal-bench/terminal-bench) task evaluation. Each task runs in an isolated container where the agent executes shell commands to solve Linux administration and DevOps challenges.

## Backends

| Backend | Description | Install |
|---|---|---|
| `"docker"` (default) | Local Docker via Harbor's `DockerEnvironment` | `pip install strands-env[terminal-bench]` |
| `"e2b"` | [e2b](https://e2b.dev) remote microVM sandbox (uses pre-baked templates) | `pip install strands-env[terminal-bench,e2b]` |

## Setup

1. **Docker** (for `"docker"` backend) — Must be installed and running:
   ```bash
   docker info  # verify Docker is available
   ```

2. **Install additional dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Task data** — Each task requires a directory with:
   ```
   task_dir/
   ├── Dockerfile          # or environment/ dir
   ├── tests/
   │   └── test.sh         # verification script
   └── environment/        # files copied into container
   ```

## Usage

```python
from strands_env.environments.terminal_bench import TerminalBenchEnv

env = TerminalBenchEnv(
    model_factory=model_factory,
    task_id="task-001",
    task_dir="/path/to/task",
    trial_dir="/path/to/output",
    timeout=1200,
)

await env.reset()       # Build and start container
result = await env.step(action)  # action.message = task.instruction
await env.cleanup()     # Stop and delete container
```

### e2b Backend

Requires `E2B_API_KEY` + `E2B_DOMAIN` env vars and a pre-baked templates JSON
mapping `task_name → template_id` (templates are built out-of-band via the
older `/templates` route on self-hosted e2b clusters).

```python
env = TerminalBenchEnv(
    model_factory=model_factory,
    task_id="task-001",
    task_dir="/path/to/task",
    trial_dir="/path/to/output",
    backend="e2b",
    e2b_backend_config={
        "templates_json": "/path/to/templates.json",   # or set E2B_PREBAKED_TEMPLATES_PATH
    },
)
```

## Tools

- **execute_command** — Execute any shell command inside the container.

## Reward

Built-in `TerminalBenchReward` (binary 0/1):
1. Uploads `tests/` to the container
2. Runs `test.sh`
3. Parses `reward.txt` output — returns 1.0 if the value is >= 1, else 0.0

Supply a custom `reward_fn` to override.

## System Prompt

The agent follows a structured problem-solving loop: analyze state, plan next steps, execute commands, and verify results.
