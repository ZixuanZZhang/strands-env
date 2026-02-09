"""Example environment hook for math reasoning evaluation with `CalculatorEnv`."""

from strands_env.cli.config import EnvConfig
from strands_env.core.models import ModelFactory
from strands_env.environments import CalculatorEnv
from strands_env.rewards import MathRewardFunction


def create_env_factory(model_factory: ModelFactory, env_config: EnvConfig):
    """Create env_factory for CalculatorEnv.

    Args:
        model_factory: Model factory provided by CLI.
        env_config: Environment configuration from CLI.

    Returns:
        Async env_factory function.
    """
    reward_fn = MathRewardFunction()

    async def env_factory(_action):
        return CalculatorEnv(
            model_factory=model_factory,
            reward_fn=reward_fn,
            system_prompt=env_config.system_prompt,
            max_tool_iterations=env_config.max_tool_iterations,
            verbose=env_config.verbose,
        )

    return env_factory
