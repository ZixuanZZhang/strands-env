# strands-env

[![CI](https://github.com/horizon-rl/strands-env/actions/workflows/test.yml/badge.svg)](https://github.com/horizon-rl/strands-env/actions/workflows/test.yml)
[![PyPI](https://img.shields.io/pypi/v/strands-env.svg)](https://pypi.org/project/strands-env/)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)

RL environment abstraction for [Strands Agents](https://github.com/strands-agents/sdk-python) — step, observe, reward.

## Features

This package standardizes agent environments by treating each `env.step()` as a full agent loop (`prompt → (tool_call, tool_response)* → response`), not a single model call. Built on [strands](https://github.com/strands-agents/sdk-python) agent loop and [`strands-sglang`](https://github.com/horizon-rl/strands-sglang) for RL training.

- **Define environments easily** — subclass `Environment` and implement tools as `@tool` functions
- **Capture token-level observations** — token-in/token-out trajectories for on-policy RL training (SGLang backend)
- **Plug in reward functions** — evaluate agent outputs with custom `RewardFunction`
- **Run benchmarks** — `Evaluator` with pass@k metrics, checkpointing, and resume

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

See [`examples/math_env.py`](examples/math_env.py) for a complete example:

```bash
python examples/math_env.py --backend sglang --sglang-base-url http://localhost:30000
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

The `Evaluator` orchestrates concurrent rollouts with checkpointing and pass@k metrics. It takes an async `env_factory` for flexible environment creation per sample, and subclasses implement `load_dataset` for different benchmarks:

```python
...
from strands_env.eval import Evaluator

class YourEvaluator(Evaluator):
    benchmark_name = "YourBenchmark"

    def load_dataset(self) -> Iterable[Action]:
        ...

async def env_factory(action: Action) -> Environment:
    ...

evaluator = YourEvaluator(
    env_factory=env_factory,
    n_samples_per_prompt=8,
    max_concurrency=30,
    keep_tokens=False, # Set True if requiring token-level trajectories (SGLang only)
    metrics_fns=[...], # Define more metrics, pass@k has been included by default
)

actions = evaluator.load_dataset()
results = await evaluator.run(actions)
metrics = evaluator.compute_metrics(results)  # {"pass@1": 0.75, "pass@8": 0.95}
```

See [`examples/aime_eval.py`](examples/aime_eval.py) for a complete example:

```bash
python examples/aime_eval.py --backend sglang --sglang-base-url http://localhost:30000
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
