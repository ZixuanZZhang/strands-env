# strands-env

[![CI](https://github.com/horizon-rl/strands-env/actions/workflows/test.yml/badge.svg)](https://github.com/horizon-rl/strands-env/actions/workflows/test.yml)
[![PyPI](https://img.shields.io/pypi/v/strands-env.svg)](https://pypi.org/project/strands-env/)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)

Standardizing environment infrastructure with [Strands Agents](https://github.com/strands-agents/sdk-python) — step, observe, reward.

> `strands-agents` is designed for serving, not training. `strands-env` integrates [`strands-sglang`](https://github.com/horizon-rl/strands-sglang) to bridge this gap.

## Define an environment

Subclass `Environment` and customize your tools:

```python
from strands_tools import calculator
from strands_env.core.environment import Environment

class MathEnv(Environment):
    def get_tools(self):
        return [calculator]
```

## Run it

```python
env = MathEnv(model_factory=factory, reward_fn=reward_fn)
result = await env.step(Action(message="What is 2^10?", task_context=TaskContext(ground_truth="1024")))

result.observation.final_response   # "1024"
result.observation.tokens           # TokenObservation (SGLang only, for on-policy RL training)
result.reward.reward                # 1.0
result.termination_reason           # task_complete
```

Each `step()` runs a full agent loop (reasoning + tool calls), not a single model call. Strands' hook-based design makes it easy to customize what happens within each step.

## Install

```bash
pip install strands-env
```

For development:

```bash
git clone https://github.com/horizon-rl/strands-env.git && cd strands-env
pip install -e ".[dev]"
```

See [`examples/math_env.py`](examples/math_env.py) for a complete runnable example:

```bash
python examples/math_env.py --backend sglang --sglang-base-url http://localhost:30000
python examples/math_env.py --backend bedrock --model-id us.anthropic.claude-sonnet-4-20250514
```

## Development

```bash
# Lint
ruff check src/ && ruff format --check src/

# Unit tests
pytest tests/unit/ -v

# Integration tests (requires running SGLang server)
pytest tests/integration/ -v --sglang-base-url=http://localhost:30000
```

## License

Apache License 2.0 - see [LICENSE](LICENSE).
