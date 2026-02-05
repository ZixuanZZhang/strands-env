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

"""AIME evaluation example — run pass@k evaluation on AIME math problems.

Usage:
    # SGLang backend (requires a running SGLang server)
    python examples/aime_eval.py --backend sglang --sglang-base-url http://localhost:30000

    # Bedrock backend (requires AWS credentials)
    python examples/aime_eval.py --backend bedrock --model-id us.anthropic.claude-sonnet-4-20250514

    # With multiple rollouts for pass@k
    python examples/aime_eval.py --backend sglang --n-rollouts 8 --k-values 1,5,8
"""

from __future__ import annotations

import asyncio
import logging

import click
import httpx

from strands_env.core.models import ModelFactory, bedrock_model_factory, sglang_model_factory
from strands_env.environments.simple_math_env import SimpleMathEnv
from strands_env.eval import AIMEEvaluator
from strands_env.rewards.math_reward import MathRewardFunction

SAMPLING_PARAMS = {"max_new_tokens": 16384, "temperature": 0.7, "top_p": 0.95}

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
    return sglang_model_factory(
        model_id=model_id,
        tokenizer=tokenizer,
        client=client,
        sampling_params=SAMPLING_PARAMS,
    )


def _create_bedrock_factory(model_id: str | None) -> ModelFactory:
    import boto3

    model_id = model_id or "us.anthropic.claude-sonnet-4-20250514"
    click.echo(f"Using Bedrock model: {model_id}")
    return bedrock_model_factory(
        model_id=model_id,
        boto_session=boto3.Session(),
        sampling_params=SAMPLING_PARAMS,
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


@click.command()
@click.option("--backend", required=True, type=click.Choice(["sglang", "bedrock"]), help="Model backend")
@click.option("--model-id", default=None, help="Model ID (auto-detected for SGLang)")
@click.option("--sglang-base-url", default="http://localhost:30000", help="SGLang server URL")
@click.option("--aime-version", default="2024", type=click.Choice(["2024", "2025"]), help="AIME dataset version")
@click.option("--n-rollouts", default=1, type=int, help="Number of rollouts per problem")
@click.option("--max-concurrency", default=10, type=int, help="Max concurrent evaluations")
@click.option("--output", default="aime_results.jsonl", help="Output file for results")
def main(
    backend: str,
    model_id: str | None,
    sglang_base_url: str,
    aime_version: str,
    n_rollouts: int,
    max_concurrency: int,
    output: str,
) -> None:
    """Run pass@k evaluation on AIME math problems."""
    # Suppress noisy library logging, keep strands_env at INFO for progress
    logging.basicConfig(level=logging.WARNING, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    logging.getLogger("strands_env").setLevel(logging.INFO)

    asyncio.run(
        run_eval(
            backend=backend,
            model_id=model_id,
            sglang_base_url=sglang_base_url,
            aime_version=aime_version,
            n_rollouts=n_rollouts,
            max_concurrency=max_concurrency,
            output=output,
        )
    )


async def run_eval(
    backend: str,
    model_id: str | None,
    sglang_base_url: str,
    aime_version: str,
    n_rollouts: int,
    max_concurrency: int,
    output: str,
) -> None:
    model_factory = create_model_factory(backend, model_id, sglang_base_url)
    reward_fn = MathRewardFunction()

    async def env_factory(_):
        env = SimpleMathEnv(model_factory=model_factory, reward_fn=reward_fn, verbose=False)
        env.system_prompt = None
        env.get_tools = lambda: []
        return env

    evaluator = AIMEEvaluator(
        env_factory=env_factory,
        n_rollouts=n_rollouts,
        max_concurrency=max_concurrency,
        output_path=output,
    )

    click.echo(f"Loading AIME {aime_version} dataset...")
    actions = evaluator.load_dataset(aime_version)
    click.echo(f"Loaded {len(actions)} problems")

    click.echo(f"Running evaluation with {n_rollouts} rollout(s) per problem...")
    results = await evaluator.run(actions)

    click.echo(f"\nResults saved to: {output}")
    click.echo(f"Total samples: {sum(len(samples) for samples in results.values())}")

    pass_at_k = evaluator.compute_pass_at_k(results, k_values=list(range(1, n_rollouts + 1)))
    click.echo(f"\npass@k metrics given (n={n_rollouts}):")
    for k, score in pass_at_k.items():
        click.echo(f"  pass@{k}: {score:.4f}")


if __name__ == "__main__":
    main()
