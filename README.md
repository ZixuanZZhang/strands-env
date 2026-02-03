# strands-env

**RL Environment Abstraction for Strands Agents — Step, Observe, Reward.**

`strands-env` defines a standard Environment interface built on [`strands-agents`](https://github.com/strands-agents/sdk-python):

```python
env = Environment(model_factory=factory, reward_fn=reward_fn)
action = Action(message="What is 2^10?", task_context=TaskContext(ground_truth="1024"))
result = await env.step(action)

print(result.observation.messages)        # Strands-format messages
print(result.observation.tokens)          # Token-level data (SGLang only for on-policy RL training)
print(result.reward)                      # RewardResult(reward=1.0, info={...})
print(result.termination_reason)          # task_complete, max_tokens_reached, ...
```

A single `env.step()` runs one full agent loop — not just a model call or tool call as in other agent environment packages. Strands' hook-based design makes it easy to customize what happens within each step.

> `strands-agents` is designed for serving, not training. `strands-env` integrates [`strands-sglang`](https://github.com/horizon-rl/strands-sglang) to bridge this gap.

## Core Concepts

| Concept | Description |
|---------|-------------|
| *Action* | Carries a user message with optional `TaskContext` (ground truth, conversation history, arbitrary metadata) |
| *Observation* | Messages from a single step, plus metrics (token usage, tool stats); for training, `TokenObservation` provides token-level data (TITO) |
| *TerminationReason* | How the step ended (completed, max tool iterations, max tokens, timeout, other errors) |
| *RewardResult* | Reward value with optional metadata |
| *StepResult* | Bundles observation, reward, and termination reason |
| *Environment* | Core abstraction — subclass and override `get_tools()` for minimal environment implementation |

## Quick Start

```bash
pip install strands-env
```

For development:

```bash
git clone <repo-url> && cd strands-env
pip install -e ".[dev]"
```

Run the simple math environment (with a calculator tool) example:

```bash
# SGLang (requires a running server)
python examples/math_env.py --backend sglang --sglang-base-url http://localhost:30000

# Bedrock (requires AWS credentials)
python examples/math_env.py --backend bedrock --model-id us.anthropic.claude-sonnet-4-20250514
```

To create your own environment on-the-fly:

1. Inherit from the base `Environment` class
2. Implement `get_tools()` with Python functions decorated with `@tool` (`from strands import tool`)
3. Run `await env.step()`

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
