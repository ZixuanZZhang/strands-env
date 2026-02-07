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

"""Benchmark registry for discovering and running evaluators."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .evaluator import Evaluator

# Registry: benchmark name -> Evaluator subclass
_BENCHMARKS: dict[str, type[Evaluator]] = {}


def register(name: str):
    """Decorator to register a benchmark evaluator.

    Example:
        @register("aime")
        class AIMEEvaluator(Evaluator):
            ...
    """

    def decorator(cls: type[Evaluator]) -> type[Evaluator]:
        if name in _BENCHMARKS:
            raise ValueError(f"Benchmark '{name}' is already registered")
        _BENCHMARKS[name] = cls
        return cls

    return decorator


def get_benchmark(name: str) -> type[Evaluator]:
    """Get a registered benchmark evaluator by name.

    Args:
        name: Benchmark name (e.g., "aime").

    Returns:
        Evaluator subclass.

    Raises:
        KeyError: If benchmark is not registered.
    """
    if name not in _BENCHMARKS:
        available = ", ".join(sorted(_BENCHMARKS.keys())) or "(none)"
        raise KeyError(f"Unknown benchmark '{name}'. Available: {available}")
    return _BENCHMARKS[name]


def list_benchmarks() -> list[str]:
    """List all registered benchmark names."""
    return sorted(_BENCHMARKS.keys())
