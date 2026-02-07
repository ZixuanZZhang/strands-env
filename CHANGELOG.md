# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- **CLI** (`strands-env`)
  - `strands-env list`: List registered benchmarks.
  - `strands-env eval <benchmark> --env <hook_file>`: Run benchmark evaluation.
  - Hook file pattern: Python files exporting `create_env_factory(model_factory, env_config)`.
  - Support for `--backend sglang|bedrock`, `--profile`, `--role-arn`, and sampling options.
  - SGLang server health check with clear error messages.
- **Benchmark Registry**
  - `@register(name)` decorator for registering benchmark evaluators.
  - `get_benchmark(name)` and `list_benchmarks()` for discovery.
  - `AIME2024Evaluator` and `AIME2025Evaluator` as separate registered benchmarks.
- **Code Quality**
  - `@override` decorator from `typing_extensions` for explicit method overrides.

### Changed

- Reorganized examples: removed `aime_eval.py` and `common.py`, added `calculator_demo.py`.
- Hook files moved to `examples/envs/`.

## [0.1.1] - 2026-02-06

### Added

- **Environments**
  - `CalculatorEnv`: Simple calculator tool for math problems.
  - `CodeSandboxEnv`: AWS Bedrock AgentCore Code Interpreter with `CodeMode` enum.
- **Evaluation**
  - `Evaluator`: Concurrent evaluation with checkpointing, resume, and pass@k metrics.
  - `AIMEEvaluator`: AIME benchmark evaluator.
  - `MathRewardFunction`: Math reward using `math-verify` for symbolic equivalence.
- **Utilities**
  - `utils/sglang.py`: SGLang client caching with `lru_cache`.
  - `utils/aws.py`: AWS boto3 session caching with `RefreshableCredentials` for auto-refresh.
- **Tools**
  - `CodeInterpreterToolkit`: `execute_code` and `execute_command` for sandboxed execution.

## [0.1.0] - 2026-02-03

Initial release with core abstractions.

- **`Environment`** base class: `step()`, `reset()`, `cleanup()`, `get_tools()`, `get_hooks()`.
- **`Action` / `TaskContext`**: User message + ground truth, conversation history, and arbitrary metadata.
- **`Observation`**: Step messages, metrics, and optional `TokenObservation` for TITO training.
- **`StepResult`**: Bundles observation, reward, and termination reason.
- **`TerminationReason`**: Maps agent exceptions to enum values via cause-chain walking.
- **`RewardFunction` / `RewardResult`**: Abstract reward interface with scalar reward + diagnostics.
- **`ModelFactory`**: Factory functions for SGLang, Bedrock, and OpenAI backends.
