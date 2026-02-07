# strands-env

[![CI](https://github.com/horizon-rl/strands-env/actions/workflows/test.yml/badge.svg)](https://github.com/horizon-rl/strands-env/actions/workflows/test.yml)
[![PyPI](https://img.shields.io/pypi/v/strands-env.svg)](https://pypi.org/project/strands-env/)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)

Standardizing environment infrastructure with [Strands Agents](https://github.com/strands-agents/sdk-python) — step, observe, reward.

## Features

This package standardizes agent environments by treating each `env.step()` as a **full agent loop**, not a single model call or tool call. Built on [strands](https://github.com/strands-agents/sdk-python) agent loop and [`strands-sglang`](https://github.com/horizon-rl/strands-sglang) for RL training.

- **Define environments easily** — subclass `Environment` and implement tools as `@tool` functions
- **Capture token-level observations** — token-in/token-out trajectories for on-policy RL training (SGLang backend)
- **Plug in reward functions** — evaluate agent outputs with custom `RewardFunction`
- **Run benchmarks** — `Evaluator` with flexible environment setup, metric customization, and resume

> An agent loop can be defined as `(prompt → (tool_call, tool_response+)* → response)`

## Install

```bash
pip install strands-env
```

For development:

```bash
git clone https://github.com/horizon-rl/strands-env.git && cd strands-env
pip install -e ".[dev]"
```

## Usage

### Define an Environment

Subclass `Environment` and add tools as `@tool`-decorated functions:

```python
from strands import tool
from strands_env.core import Environment

@tool
def calculator(expression: str) -> str:
    """Evaluate a math expression."""
    return str(eval(expression))

class MathEnv(Environment):
    def get_tools(self):
        return [calculator]
```

### Run It

```python
env = MathEnv(model_factory=factory, reward_fn=reward_fn)
result = await env.step(Action(message="What is 2^10?", task_context=TaskContext(ground_truth="1024")))

result.observation.final_response   # "1024"
result.observation.tokens           # TokenObservation (SGLang only)
result.reward.reward                # 1.0
result.termination_reason           # TerminationReason.TASK_COMPLETE
```

See [`examples/calculator_demo.py`](examples/calculator_demo.py) for a complete example:

```bash
python examples/calculator_demo.py --backend sglang --base-url http://localhost:30000
```

## RL Training

For RL training with [slime](https://github.com/THUDM/slime/), customize the `generate` and `reward_func` methods to replace single generation with agentic rollout:

```python
from strands_env.core import Action, TaskContext
from strands_env.core.models import sglang_model_factory
from strands_env.utils import get_cached_client_from_slime_args

async def generate(args, sample, sampling_params):
    # Build model factory with cached client
    factory = sglang_model_factory(
        model_id=args.hf_checkpoint,
        tokenizer=tokenizer,
        client=get_cached_client_from_slime_args(args),
        sampling_params=sampling_params,
    )

    # Create environment and run step
    env = YourEnv(model_factory=factory, reward_fn=None)
    action = Action(message=sample.prompt, task_context=TaskContext(ground_truth=sample.label))
    step_result = await env.step(action)

    # Extract TITO data for training
    token_obs = step_result.observation.tokens
    sample.tokens = token_obs.token_ids
    sample.loss_mask = token_obs.rollout_loss_mask
    sample.rollout_log_probs = token_obs.rollout_logprobs
    sample.response_length = len(token_obs.rollout_token_ids)

    # Attach for reward computation
    sample.action = action
    sample.step_result = step_result
    return sample

async def reward_func(args, sample, **kwargs):
    reward_fn = YourRewardFunction()
    reward_result = await reward_fn.compute(action=sample.action, step_result=sample.step_result)
    return reward_result.reward
```

Key points:
- `get_cached_client_from_slime_args(args)` provides connection pooling across rollouts
- `TokenObservation` contains token IDs and logprobs for on-policy training
- Reward is computed separately to allow async/batched reward computation

## Evaluation

### CLI

The `strands-env` CLI provides commands for running benchmark evaluations:

```bash
# List available benchmarks
strands-env list

# Run AIME 2024 evaluation with SGLang
strands-env eval aime-2024 --env examples/envs/calculator_env.py --backend sglang

# Run with Bedrock
strands-env eval aime-2024 --env examples/envs/code_sandbox_env.py --backend bedrock --model-id us.anthropic.claude-sonnet-4-20250514

# With multiple samples for pass@k
strands-env eval aime-2024 --env examples/envs/calculator_env.py --backend sglang --n-samples 8 --max-concurrency 30
```

### Hook Files

Environment hook files define how environments are created. They export a `create_env_factory` function:

```python
# examples/envs/calculator_env.py
from strands_env.cli.config import EnvConfig
from strands_env.core.models import ModelFactory
from strands_env.environments.calculator import CalculatorEnv
from strands_env.rewards.math_reward import MathRewardFunction

def create_env_factory(model_factory: ModelFactory, env_config: EnvConfig):
    reward_fn = MathRewardFunction()

    async def env_factory(_action):
        return CalculatorEnv(
            model_factory=model_factory,
            reward_fn=reward_fn,
            system_prompt=env_config.system_prompt,
            max_tool_iterations=env_config.max_tool_iterations,
        )

    return env_factory
```

### Programmatic Usage

For custom evaluators, subclass `Evaluator` and implement `load_dataset`:

```python
from strands_env.eval import Evaluator, register

@register("my-benchmark")
class MyEvaluator(Evaluator):
    benchmark_name = "my-benchmark"

    def load_dataset(self) -> Iterable[Action]:
        ...
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

Apache License 2.0 — see [LICENSE](LICENSE).
