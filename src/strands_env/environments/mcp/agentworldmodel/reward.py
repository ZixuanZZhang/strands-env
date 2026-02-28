"""Reward function for AgentWorldModel tasks.

Executes the per-task ``verify_task_completion`` function against final answers
or SQLite database state changes to produce a binary reward.
"""

from __future__ import annotations

import json
import logging
import sqlite3

from strands_env.core.types import Action, RewardFunction, RewardResult, StepResult

logger = logging.getLogger(__name__)


def _extract_final_response(messages: list) -> str:
    """Return the text content of the last assistant message, or empty string."""
    for msg in reversed(messages):
        if msg.get("role") == "assistant":
            content = msg.get("content", "")
            if isinstance(content, list):
                return "".join(block.get("text", "") for block in content if "text" in block)
            return str(content)
    return ""


class AWMRewardFunction(RewardFunction):
    """Run AWM's execution-based verification codes and return binary reward."""

    async def compute(self, action: Action, step_result: StepResult) -> RewardResult:
        ctx = action.task_context
        final_answer = _extract_final_response(step_result.observation.messages)

        try:
            namespace: dict = {"sqlite3": sqlite3, "json": json}
            exec(ctx.verify_code, namespace)  # noqa: S102
            result = namespace["verify_task_completion"](
                initial_db_path=ctx.initial_db_path,
                final_db_path=ctx.work_db_path,
                final_answer=final_answer,
            )
        except Exception as e:
            logger.warning("Verification failed for %s task %s: %s", ctx.scenario, ctx.task_idx, e)
            return RewardResult(reward=0.0, info={"error": str(e)})

        reward = 1.0 if isinstance(result, dict) and result.get("result") == "complete" else 0.0
        logger.info("Verification %s task %d: %s", ctx.scenario, ctx.task_idx, result)
        return RewardResult(reward=reward, info={"verification_result": result})
