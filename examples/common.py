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

"""Common configuration for example scripts.

Modify the defaults below to customize model behavior across all examples.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Literal

import httpx

from strands_env.core.models import ModelFactory, bedrock_model_factory, sglang_model_factory
from strands_env.utils.sglang import get_cached_client

# Suppress transformers warning about missing PyTorch/TensorFlow
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")

# ---------------------------------------------------------------------------
# Default Configuration (modify these values)
# ---------------------------------------------------------------------------

# SGLang server URL
DEFAULT_SGLANG_URL = "http://localhost:30000"

# Default Bedrock model ID
DEFAULT_BEDROCK_MODEL = "us.anthropic.claude-sonnet-4-20250514-v1:0"

# Sampling parameters
DEFAULT_MAX_NEW_TOKENS = 16384
DEFAULT_TEMPERATURE = 0.7
DEFAULT_TOP_P = 0.95
DEFAULT_TOP_K = None  # Set to int to enable top-k sampling

# ---------------------------------------------------------------------------
# Configuration Classes
# ---------------------------------------------------------------------------


@dataclass
class SamplingParams:
    """Sampling parameters for model generation."""

    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS
    temperature: float = DEFAULT_TEMPERATURE
    top_p: float = DEFAULT_TOP_P
    top_k: int | None = DEFAULT_TOP_K

    def to_dict(self) -> dict:
        """Convert to dict, excluding None values."""
        return {k: v for k, v in self.__dict__.items() if v is not None}


@dataclass
class ModelConfig:
    """Model configuration for creating model factories."""

    backend: Literal["sglang", "bedrock"]
    model_id: str | None = None
    base_url: str = DEFAULT_SGLANG_URL
    sampling_params: SamplingParams = field(default_factory=SamplingParams)

    def create_factory(self) -> ModelFactory:
        """Create a ModelFactory based on the configuration."""
        if self.backend == "sglang":
            return self._create_sglang_factory()
        elif self.backend == "bedrock":
            return self._create_bedrock_factory()
        else:
            raise ValueError(f"Unknown backend: {self.backend}")

    def _create_sglang_factory(self) -> ModelFactory:
        from transformers import AutoTokenizer

        base_url = self.base_url.rstrip("/")
        model_id = self.model_id

        # Auto-detect model ID from server if not provided
        if model_id is None:
            resp = httpx.get(f"{base_url}/get_model_info", timeout=10)
            resp.raise_for_status()
            model_id = resp.json()["model_path"]

        tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        client = get_cached_client(base_url, max_connections=1000)
        return sglang_model_factory(
            model_id=model_id,
            tokenizer=tokenizer,
            client=client,
            sampling_params=self.sampling_params.to_dict(),
        )

    def _create_bedrock_factory(self) -> ModelFactory:
        import boto3

        model_id = self.model_id or DEFAULT_BEDROCK_MODEL
        return bedrock_model_factory(
            model_id=model_id,
            boto_session=boto3.Session(),
            sampling_params=self.sampling_params.to_dict(),
        )
