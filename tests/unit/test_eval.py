"""Unit tests for evaluation module."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from strands_env.core import Action, Environment, Observation, RewardResult, StepResult, TaskContext
from strands_env.eval import EvalSample, Evaluator

# ---------------------------------------------------------------------------
# EvalSample
# ---------------------------------------------------------------------------


class TestEvalSample:
    def test_basic_fields(self):
        step_result = StepResult(observation=Observation())
        sample = EvalSample(action=Action(message="test"), step_result=step_result)
        assert sample.action.message == "test"
        assert sample.step_result == step_result


# ---------------------------------------------------------------------------
# Evaluator
# ---------------------------------------------------------------------------


class TestEvaluator:
    @pytest.fixture
    def mock_env(self):
        env = MagicMock(spec=Environment)
        env.reset = AsyncMock()
        env.step = AsyncMock()
        env.cleanup = AsyncMock()
        return env

    async def test_factory_mode(self, mock_env, tmp_path):
        """Factory mode: reset/step/cleanup called for each sample."""
        mock_env.step.return_value = StepResult(observation=Observation())

        async def factory(action):
            return mock_env

        actions = [Action(message=f"q{i}", task_context=TaskContext(id=f"p{i}")) for i in range(3)]

        evaluator = Evaluator(env_factory=factory, output_path=tmp_path / "results.jsonl")
        results = await evaluator.run(actions)

        # With n_rollouts=1 (default), each action is run once
        assert mock_env.reset.await_count == 3
        assert mock_env.step.await_count == 3
        assert mock_env.cleanup.await_count == 3
        assert len(results) == 3  # 3 problem_ids
        assert sum(len(samples) for samples in results.values()) == 3

    async def test_n_rollouts_duplication(self, mock_env, tmp_path):
        """Each action is duplicated n_rollouts times."""
        mock_env.step.return_value = StepResult(observation=Observation())

        async def factory(action):
            return mock_env

        actions = [Action(message="q", task_context=TaskContext(id="p1"))]

        evaluator = Evaluator(env_factory=factory, n_rollouts=5, output_path=tmp_path / "results.jsonl")
        results = await evaluator.run(actions)

        # 5 rollouts per problem
        assert mock_env.step.await_count == 5
        assert len(results) == 1  # One problem_id key
        assert "p1" in results
        assert len(results["p1"]) == 5  # 5 samples for that problem

        # Each should have unique sample_id (stored in action.task_context.id)
        sample_ids = [s.action.task_context.id for s in results["p1"]]
        assert len(set(sample_ids)) == 5

    async def test_factory_receives_action(self, tmp_path):
        """Factory receives the action for per-sample configuration."""
        received_actions = []

        async def factory(action):
            received_actions.append(action)
            env = MagicMock()
            env.reset = AsyncMock()
            env.step = AsyncMock(return_value=StepResult(observation=Observation()))
            env.cleanup = AsyncMock()
            return env

        actions = [Action(message="a"), Action(message="b")]
        evaluator = Evaluator(env_factory=factory, output_path=tmp_path / "results.jsonl")
        await evaluator.run(actions)

        assert len(received_actions) == 2
        assert received_actions[0].message == "a"
        assert received_actions[1].message == "b"

    async def test_concurrency_control(self, tmp_path):
        """Verify semaphore limits concurrent calls."""
        import asyncio

        concurrent_count = 0
        max_concurrent = 0

        async def mock_step(action):
            nonlocal concurrent_count, max_concurrent
            concurrent_count += 1
            max_concurrent = max(max_concurrent, concurrent_count)
            await asyncio.sleep(0.01)
            concurrent_count -= 1
            return StepResult(observation=Observation())

        async def factory(action):
            env = MagicMock()
            env.reset = AsyncMock()
            env.step = mock_step
            env.cleanup = AsyncMock()
            return env

        actions = [Action(message=f"q{i}") for i in range(10)]

        evaluator = Evaluator(env_factory=factory, max_concurrency=3, output_path=tmp_path / "results.jsonl")
        await evaluator.run(actions)

        assert max_concurrent <= 3

    async def test_empty_actions(self, mock_env, tmp_path):
        """Empty actions produces empty results."""

        async def factory(action):
            return mock_env

        evaluator = Evaluator(env_factory=factory, output_path=tmp_path / "results.jsonl")
        results = await evaluator.run([])

        assert results == {}
        mock_env.step.assert_not_awaited()


# ---------------------------------------------------------------------------
# Checkpoint/Resume
# ---------------------------------------------------------------------------


class TestCheckpoint:
    @pytest.fixture
    def mock_env(self):
        env = MagicMock(spec=Environment)
        env.reset = AsyncMock()
        env.step = AsyncMock()
        env.cleanup = AsyncMock()
        return env

    async def test_saves_checkpoint(self, mock_env, tmp_path):
        """Results saved to output_path."""
        mock_env.step.return_value = StepResult(observation=Observation())
        output_path = tmp_path / "results.jsonl"

        async def factory(action):
            return mock_env

        evaluator = Evaluator(env_factory=factory, output_path=output_path, save_interval=1)
        await evaluator.run([Action(message="q1", task_context=TaskContext(id="s1"))])

        assert output_path.exists()
        lines = output_path.read_text().strip().split("\n")
        assert len(lines) == 1

    async def test_resumes_from_checkpoint(self, mock_env, tmp_path):
        """Skips already-completed samples on resume."""
        mock_env.step.return_value = StepResult(observation=Observation())
        output_path = tmp_path / "results.jsonl"

        async def factory(action):
            return mock_env

        # First run - complete s1
        evaluator1 = Evaluator(env_factory=factory, output_path=output_path, save_interval=1)
        await evaluator1.run([Action(message="q1", task_context=TaskContext(id="s1"))])
        assert mock_env.step.await_count == 1

        # Second run - s1 skipped, s2 processed
        mock_env.step.reset_mock()
        evaluator2 = Evaluator(env_factory=factory, output_path=output_path, save_interval=1)
        results = await evaluator2.run([
            Action(message="q1", task_context=TaskContext(id="s1")),
            Action(message="q2", task_context=TaskContext(id="s2")),
        ])

        assert mock_env.step.await_count == 1  # Only s2 was processed
        assert len(results) == 2  # Both problem_ids in results
        assert sum(len(samples) for samples in results.values()) == 2


# ---------------------------------------------------------------------------
# pass@k
# ---------------------------------------------------------------------------


class TestPassAtKSingle:
    def test_all_correct(self):
        assert Evaluator._pass_at_k_single(n=10, c=10, k=1) == 1.0
        assert Evaluator._pass_at_k_single(n=10, c=10, k=5) == 1.0

    def test_none_correct(self):
        assert Evaluator._pass_at_k_single(n=10, c=0, k=1) == 0.0
        assert Evaluator._pass_at_k_single(n=10, c=0, k=5) == 0.0

    def test_half_correct(self):
        # pass@1 = 1 - C(5,1)/C(10,1) = 1 - 5/10 = 0.5
        assert Evaluator._pass_at_k_single(n=10, c=5, k=1) == pytest.approx(0.5)

    def test_one_correct_out_of_ten(self):
        # pass@1 = 1 - 9/10 = 0.1
        assert Evaluator._pass_at_k_single(n=10, c=1, k=1) == pytest.approx(0.1)
        # pass@5 = 1 - C(9,5)/C(10,5) = 0.5
        assert Evaluator._pass_at_k_single(n=10, c=1, k=5) == pytest.approx(0.5)

    def test_not_enough_incorrect(self):
        # If n - c < k, pass@k = 1.0 (guaranteed to get a correct one)
        assert Evaluator._pass_at_k_single(n=5, c=4, k=2) == 1.0


class TestComputePassAtK:
    def _make_sample(self, reward: float, idx: int = 0) -> EvalSample:
        return EvalSample(
            action=Action(message="q", task_context=TaskContext(id=f"sample_{idx}")),
            step_result=StepResult(observation=Observation(), reward=RewardResult(reward=reward)),
        )

    def test_empty_samples(self):
        result = Evaluator.compute_pass_at_k({}, k_values=[1])
        assert result == {1: 0.0}

    def test_all_correct(self):
        results = {"p1": [self._make_sample(1.0, i) for i in range(5)]}
        result = Evaluator.compute_pass_at_k(results, k_values=[1])
        assert result[1] == 1.0

    def test_none_correct(self):
        results = {"p1": [self._make_sample(0.0, i) for i in range(5)]}
        result = Evaluator.compute_pass_at_k(results, k_values=[1])
        assert result[1] == 0.0

    def test_multiple_problems(self):
        # p1: 2/2 correct -> pass@1 = 1.0
        # p2: 0/2 correct -> pass@1 = 0.0
        results = {
            "p1": [self._make_sample(1.0, 0), self._make_sample(1.0, 1)],
            "p2": [self._make_sample(0.0, 0), self._make_sample(0.0, 1)],
        }
        # Average: (1.0 + 0.0) / 2 = 0.5
        result = Evaluator.compute_pass_at_k(results, k_values=[1])
        assert result[1] == pytest.approx(0.5)

    def test_multiple_k_values(self):
        # p1: 1/5 correct
        results = {"p1": [self._make_sample(1.0 if i == 0 else 0.0, i) for i in range(5)]}
        result = Evaluator.compute_pass_at_k(results, k_values=[1, 5])
        # pass@1 = 1 - 4/5 = 0.2
        assert result[1] == pytest.approx(0.2)
        # pass@5 = 1.0 (n-c=4 < k=5, guaranteed to get the correct one)
        assert result[5] == pytest.approx(1.0)

    def test_custom_reward_threshold(self):
        results = {"p1": [self._make_sample(0.5, 0)]}
        # Default threshold 1.0 - not correct
        result = Evaluator.compute_pass_at_k(results, k_values=[1], reward_threshold=1.0)
        assert result[1] == 0.0
        # Threshold 0.5 - correct
        result = Evaluator.compute_pass_at_k(results, k_values=[1], reward_threshold=0.5)
        assert result[1] == 1.0

    def test_k_larger_than_n_skipped(self):
        # Only 2 samples, k=5 - this problem is skipped
        results = {"p1": [self._make_sample(1.0, 0), self._make_sample(1.0, 1)]}
        result = Evaluator.compute_pass_at_k(results, k_values=[5])
        assert result[5] == 0.0  # No problems have enough samples

    def test_none_reward_handled(self):
        """Samples with None reward are treated as incorrect."""
        sample = EvalSample(
            action=Action(message="q", task_context=TaskContext(id="p1_0")),
            step_result=StepResult(observation=Observation(), reward=None),
        )
        result = Evaluator.compute_pass_at_k({"p1": [sample]}, k_values=[1])
        assert result[1] == 0.0
