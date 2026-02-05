# Copyright 2025 Horizon RL Contributors

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Math environment example — run math problems through a Strands agent with a calculator tool.

Usage:
    # SGLang backend (requires a running SGLang server)
    python examples/math_env.py --backend sglang --sglang-base-url http://localhost:30000

    # Bedrock backend (requires AWS credentials)
    python examples/math_env.py --backend bedrock --model-id us.anthropic.claude-sonnet-4-20250514
"""

from __future__ import annotations

import asyncio
import logging

import click
import httpx

from strands_env.core.models import ModelFactory, bedrock_model_factory, sglang_model_factory
from strands_env.core.types import Action, TaskContext
from strands_env.environments.simple_math_env import SimpleMathEnv
from strands_env.rewards.math_reward import MathRewardFunction

logger = logging.getLogger(__name__)

MATH_PROBLEMS = [
    ("What is 123 * 456?", "56088"),
    ("What is the square root of 144?", "12"),
    ("What is 2^10?", "1024"),
]


# ---------------------------------------------------------------------------
# Model factory helpers
# ---------------------------------------------------------------------------


def create_model_factory(backend: str, model_id: str | None, sglang_base_url: str) -> ModelFactory:
    if backend == "sglang":
        return _create_sglang_factory(model_id, sglang_base_url)
    elif backend == "bedrock":
        return _create_bedrock_factory(model_id)
    else:
        raise ValueError(f"Unknown backend: {backend}")


def _create_sglang_factory(model_id: str | None, sglang_base_url: str) -> ModelFactory:
    from strands_sglang import SGLangClient
    from transformers import AutoTokenizer

    base_url = sglang_base_url.rstrip("/")
    if model_id is None:
        resp = httpx.get(f"{base_url}/get_model_info", timeout=10)
        resp.raise_for_status()
        model_id = resp.json()["model_path"]
        click.echo(f"Auto-detected model: {model_id}")

    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    client = SGLangClient(base_url)
    return sglang_model_factory(model_id=model_id, tokenizer=tokenizer, client=client)


def _create_bedrock_factory(model_id: str | None) -> ModelFactory:
    import boto3

    model_id = model_id or "us.anthropic.claude-sonnet-4-20250514"
    click.echo(f"Using Bedrock model: {model_id}")
    return bedrock_model_factory(model_id=model_id, boto_session=boto3.Session())


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


@click.command()
@click.option("--backend", required=True, type=click.Choice(["sglang", "bedrock"]), help="Model backend")
@click.option("--model-id", default=None, help="Model ID (auto-detected for SGLang)")
@click.option("--sglang-base-url", default="http://localhost:30000", help="SGLang server URL")
@click.option("--verbose", is_flag=True, help="Print agent callback output")
def main(backend: str, model_id: str | None, sglang_base_url: str, verbose: bool) -> None:
    """Run math problems through a Strands agent with a calculator tool."""
    logging.basicConfig(level=logging.WARNING)
    asyncio.run(run_math_env(backend, model_id, sglang_base_url, verbose))


async def run_math_env(backend: str, model_id: str | None, sglang_base_url: str, verbose: bool) -> None:
    model_factory = create_model_factory(backend, model_id, sglang_base_url)
    env = SimpleMathEnv(model_factory=model_factory, reward_fn=MathRewardFunction(), verbose=verbose)

    for question, ground_truth in MATH_PROBLEMS:
        click.echo(f"\n{'=' * 60}")
        click.echo(f"Question: {question}")
        click.echo(f"Expected: {ground_truth}")
        click.echo("-" * 60)

        action = Action(message=question, task_context=TaskContext(ground_truth=ground_truth))
        result = await env.step(action)

        click.echo(f"Termination: {result.termination_reason.value}")
        click.echo(f"Response:    {result.observation.final_response}")
        click.echo(f"Reward:      {result.reward.reward if result.reward else None}")
        click.echo(f"Metrics:     {result.observation.metrics}")


if __name__ == "__main__":
    main()
