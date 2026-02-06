# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- **`Evaluator`**: Concurrent evaluation orchestrator with checkpointing, resume, and pass@k metrics.
  - tqdm progress bar with `logging_redirect_tqdm` for clean output
  - `n_samples_per_prompt` for pass@k evaluation
  - JSONL checkpointing with automatic resume
- **`AIMEEvaluator`**: AIME benchmark evaluator subclass.
- **`MathRewardFunction`**: Math reward using `math-verify` for symbolic equivalence checking.
- **`utils/sglang.py`**: SGLang client caching utilities.
  - `get_cached_client(base_url, max_connections)` with `lru_cache`
  - `get_cached_client_from_slime_args(args)` for slime RL training integration
- **`utils/aws.py`**: AWS boto3 session caching utilities.
  - `get_boto3_session(region, profile_name)` with `lru_cache`
  - `get_assumed_role_session(role_arn, region)` with `RefreshableCredentials` for auto-refresh
- **`tools/code_interpreter.py`**: `CodeInterpreterToolkit` for AWS Bedrock AgentCore Code Interpreter.
  - `execute_code` tool for running Python code
  - `execute_command` tool for running shell commands
- Added `boto3`, `bedrock-agentcore`, `datasets`, `tqdm` to main dependencies.

## [0.0.2] - 2026-02-03

### Fixed

- Replace git dependency (`strands-sglang @ git+...`) with PyPI package (`strands-sglang>=0.1.2`) to fix PyPI upload rejection.

## [0.0.1] - 2026-02-03 [yanked]

Initial release — core abstractions only. Environments will be added in future releases.

### Added

- **`Environment`** base class: `step()`, `reset()`, `cleanup()`, `get_tools()`, `get_hooks()`, `compute_metrics()`.
- **`Action` / `TaskContext`**: User message + ground truth, conversation history, and arbitrary metadata (`extra="allow"`).
- **`Observation`**: Step messages, metrics, and optional `TokenObservation` for TITO training.
- **`StepResult`**: Bundles observation, reward, and termination reason.
- **`TerminationReason`**: Maps agent exceptions (`MaxToolIterationsReachedError`, `MaxTokensReachedException`, timeouts) to enum values via cause-chain walking.
- **`RewardFunction` / `RewardResult`**: Abstract reward interface with scalar reward + diagnostics.
- **`ModelFactory`** type and factory functions for SGLang, Bedrock, and OpenAI backends.
- **`examples/math_env.py`**: Calculator tool example with exact-match reward, supporting SGLang and Bedrock.
- CI/CD: GitHub Actions for testing (lint + unit tests on Python 3.10–3.12) and PyPI publishing.
