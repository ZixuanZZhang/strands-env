# Copyright 2025-2026 Strands RL Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""High-concurrency async client for managing per-task pods on EKS Fargate.

Each pod runs `envd` (see `envd.py`) as PID 1 via a ConfigMap mount, listening
on `0.0.0.0:49983`. Client RPC goes through the API server's pod proxy
(`/api/v1/.../pods/{name}:49983/proxy/{path}`), reusing the
`kubernetes_asyncio.ApiClient`'s aiohttp session, bearer token (lazy-refresh),
SSL context, and HTTP keepalive.

`/commands/run` uses chunked NDJSON streaming (`started` → `heartbeat`* →
`result`). The client raises `SandboxUnresponsive` if envd goes silent for
`HEARTBEAT_GRACE_SEC`, turning "pod hangs" into a fast-fail typed exception.

A singleton `PodWatcher` per `(cluster, namespace)` shares one Watch stream
across all pods (label-selected on `strands-env/managed=true`). Each pod gets
a ready/dead future pair; `exec()` cross-checks `dead_future` via
`asyncio.wait`, so pod death cancels the in-flight call immediately.

Exception taxonomy: `EKSPodError` (base), `PodStartError`, `SandboxDied`,
`SandboxOOMKilled` (subclass of `SandboxDied`), `SandboxNotFound`,
`SandboxUnresponsive`, `PodExecTimeoutError`.
"""

from __future__ import annotations

import asyncio
import base64
import hashlib
import io
import json
import logging
import random
import re
import socket
import tarfile
import time
import uuid
from contextlib import suppress
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

from tenacity import (
    RetryCallState,
    before_sleep_log,
    retry,
    retry_if_exception,
    stop_after_attempt,
)

UTC = timezone.utc  # `datetime.UTC` is 3.11+; we support 3.10+.

if TYPE_CHECKING:  # pragma: no cover - typing only
    import boto3
    from kubernetes_asyncio.client import ApiClient, Configuration, CoreV1Api

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

ENVD_PORT = 49983
_ENVD_MOUNT_PATH = "/etc/strands-envd"

#: How often envd emits a heartbeat. Client tolerates 3× missing before failing.
HEARTBEAT_INTERVAL_SEC = 30.0
HEARTBEAT_GRACE_SEC = 90.0

#: Backstop request timeout. Subprocess timeouts can be larger; envd closes
#: the connection via the `result` frame well before this fires.
DEFAULT_REQUEST_TIMEOUT_SEC = 7200.0
DEFAULT_POD_ACTIVE_DEADLINE_SEC = 9600

#: SO_KEEPALIVE tuning — Linux default 2h idle is useless against AWS LB /
#: VPC endpoint 350s reset. Tighten so dead peers are detected in ~3 min.
_TCP_KEEPALIVE_IDLE = 60
_TCP_KEEPALIVE_INTVL = 30
_TCP_KEEPALIVE_CNT = 4

#: Pod PID 1. Installs python3 if missing (v0.1 stopgap; proper fix is a Go
#: static binary via init container — independent PR).
_ENVD_BOOTSTRAP = (
    "set -e; "
    "if ! command -v python3 >/dev/null 2>&1; then "
    '  echo "[envd-bootstrap] python3 missing, installing..." >&2; '
    "  if command -v apt-get >/dev/null 2>&1; then "
    "    apt-get update -qq && apt-get install -y --no-install-recommends python3; "
    "  elif command -v apk >/dev/null 2>&1; then apk add --no-cache python3; "
    "  elif command -v dnf >/dev/null 2>&1; then dnf install -y python3; "
    "  elif command -v yum >/dev/null 2>&1; then yum install -y python3; "
    '  else echo "[envd-bootstrap] FATAL: no apt/apk/dnf/yum" >&2; exit 1; fi; '
    "fi; exec python3 " + _ENVD_MOUNT_PATH + "/envd.py"
)

_ENVD_SCRIPT_PATH = Path(__file__).parent / "envd.py"


def _load_envd_script() -> tuple[str, str]:
    """Return `(content, sha256_8char_hex)` of envd.py — names the ConfigMap."""
    content = _ENVD_SCRIPT_PATH.read_text(encoding="utf-8")
    digest = hashlib.sha256(content.encode("utf-8")).hexdigest()[:8]
    return content, digest


_ENVD_SCRIPT_CONTENT, _ENVD_SCRIPT_HASH = _load_envd_script()


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class EKSPodError(RuntimeError):
    """Base for all errors raised by this module."""


class PodStartError(EKSPodError):
    """Pod could not become Ready: image pull failed, unschedulable, or entered Failed."""


class SandboxDied(EKSPodError):  # noqa: N818 — `Sandbox*` taxonomy matches e2b SDK
    """Pod transitioned to a terminal state during a call (Failed, Succeeded, etc.)."""


class SandboxOOMKilled(SandboxDied):  # noqa: N818
    """Container was OOMKilled (cgroup memory exhausted). Subtype of `SandboxDied`."""


class SandboxNotFound(EKSPodError):  # noqa: N818
    """Pod no longer exists (404 from API server)."""


class SandboxUnresponsive(EKSPodError):  # noqa: N818
    """No envd frame received within `HEARTBEAT_GRACE_SEC` — envd or subprocess wedged."""


class PodExecTimeoutError(EKSPodError):
    """Subprocess exceeded its `timeout_sec` budget (envd `result` frame had `timeout=true`)."""


@dataclass
class ExecResult:
    """Result of an `exec` call."""

    stdout: str
    stderr: str
    return_code: int
    #: True if envd ring-buffer-trimmed stdout/stderr at the per-stream cap.
    #: When set, strings hold only the *tail* — earlier context is lost.
    truncated: bool = False


# ---------------------------------------------------------------------------
# Image rewriting + EKS bearer token
# ---------------------------------------------------------------------------

_REGISTRY_RE = re.compile(r"^(?:localhost(?::\d+)?|[^/]+[.:][^/]*)/")


def resolve_image(image: str, ecr_pull_through_cache: str | None) -> str:
    """Rewrite a Docker Hub reference through an ECR pull-through cache. Other registries pass through."""
    if not ecr_pull_through_cache:
        return image
    if _REGISTRY_RE.match(image):
        return image
    cache = ecr_pull_through_cache.rstrip("/")
    name = image if "/" in image else f"library/{image}"
    return f"{cache}/{name}"


def generate_eks_token(cluster_name: str, session: boto3.Session) -> tuple[str, datetime]:
    """Mint an EKS bearer token via a presigned STS URL. Same format `aws eks get-token` produces."""
    from botocore.signers import RequestSigner

    sts = session.client("sts", region_name=session.region_name)
    signer = RequestSigner(
        sts.meta.service_model.service_id,
        session.region_name,
        "sts",
        "v4",
        session.get_credentials(),
        session.events,
    )
    params = {
        "method": "GET",
        "url": f"https://sts.{session.region_name}.amazonaws.com/?Action=GetCallerIdentity&Version=2011-06-15",
        "body": {},
        "headers": {"x-k8s-aws-id": cluster_name},
        "context": {},
    }
    signed = signer.generate_presigned_url(params, region_name=session.region_name, expires_in=60, operation_name="")
    encoded = base64.urlsafe_b64encode(signed.encode("utf-8")).rstrip(b"=").decode("utf-8")
    return f"k8s-aws-v1.{encoded}", datetime.now(UTC) + timedelta(minutes=14)


# ---------------------------------------------------------------------------
# Retry policy + TCP keepalive
# ---------------------------------------------------------------------------


def _is_transient(e: BaseException) -> bool:
    """Retry K8s 429/5xx + aiohttp transport-level errors. `exec` is never retried.

    aiohttp `ClientConnectionError` covers DNS failures (`ClientConnectorDNSError`),
    TCP refusals, mid-handshake resets, and silent disconnects — all conditions
    where the request demonstrably did NOT reach envd, so retry has no
    double-execute risk. We see these in practice when CoreDNS / kube-dns
    hiccups during long eval runs (apiserver endpoint resolution fails for a
    few seconds, transparently recovers).

    `exec` is decorated separately and intentionally has no `@_retry`; this
    predicate only affects the idempotent ops (`upload_dir`, `download_dir`,
    `_api_create_pod`, `_delete_pod`, `logs`).
    """
    import aiohttp
    from kubernetes_asyncio.client.exceptions import ApiException

    if isinstance(e, ApiException) and e.status in (429, 500, 502, 503, 504):
        return True
    return isinstance(e, aiohttp.ClientConnectionError)


def _jittered_exp_wait(state: RetryCallState) -> float:
    """0.5→1→2→4→8→16 capped at 30s, ±25% relative jitter.

    Multiplicative jitter (not tenacity's additive `wait_exponential_jitter`) so
    a single jitter knob scales the whole curve — matches the AWS / Google
    client convention used elsewhere.
    """
    base = min(0.5 * (2 ** (state.attempt_number - 1)), 30.0)
    return base * random.uniform(0.75, 1.25)


_retry = retry(
    retry=retry_if_exception(_is_transient),
    stop=stop_after_attempt(5),
    wait=_jittered_exp_wait,
    before_sleep=before_sleep_log(logger, logging.WARNING),
    reraise=True,
)


def _make_keepalive_socket(addr_info: Any) -> socket.socket:
    """Aiohttp `socket_factory` enabling SO_KEEPALIVE with tight intervals.

    Linux default is 2h idle + 75s × 9 — useless against 350s LB resets. We
    tighten to 60s idle + 30s × 4 so dead peers are detected within ~3 min.
    """
    family, type_, proto, _canon, _sockaddr = addr_info
    s = socket.socket(family, type_, proto)
    s.setblocking(False)
    s.setsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)
    # TCP_KEEPIDLE/INTVL/CNT are Linux; macOS only has TCP_KEEPALIVE. Skip silently.
    for opt_name, val in (
        ("TCP_KEEPIDLE", _TCP_KEEPALIVE_IDLE),
        ("TCP_KEEPINTVL", _TCP_KEEPALIVE_INTVL),
        ("TCP_KEEPCNT", _TCP_KEEPALIVE_CNT),
    ):
        opt = getattr(socket, opt_name, None)
        if opt is not None:
            with suppress(OSError):
                s.setsockopt(socket.IPPROTO_TCP, opt, val)
    return s


# ---------------------------------------------------------------------------
# PodWatcher — singleton Watch stream + per-pod ready/dead futures
# ---------------------------------------------------------------------------

#: Container `waiting.reason` values that mean "permanent, won't recover".
_PERMANENT_WAITING_REASONS = frozenset(
    {
        "ImagePullBackOff",
        "ErrImagePull",
        "InvalidImageName",
        "ImageInspectError",
        "RegistryUnavailable",
        "CreateContainerConfigError",
        "CreateContainerError",
        "RunContainerError",
        "PreStartHookError",
        "PostStartHookError",
    }
)


@dataclass
class _Ready:
    """Sentinel: pod is Running + container ready."""


@dataclass
class _Dead:
    """Sentinel: terminal state. `error` is the typed exception to raise."""

    error: EKSPodError


class _PodEntry:
    """Per-pod state held by `PodWatcher`. Futures are created on `register`."""

    __slots__ = ("name", "ready_future", "dead_future")

    def __init__(self, name: str, loop: asyncio.AbstractEventLoop) -> None:
        """Initialize a `_PodEntry` instance with fresh futures on `loop`."""
        self.name = name
        self.ready_future: asyncio.Future[None] = loop.create_future()
        self.dead_future: asyncio.Future[EKSPodError] = loop.create_future()


class PodWatcher:
    """Singleton-per-namespace Watch stream covering all `strands-env/managed=true` pods.

    Why a singleton: a per-pod `Watch` is one long-lived apiserver stream and one
    inflight slot. At 1000 concurrent rollouts that's 1000 streams. One shared
    stream + label selector handles the same work in a single slot.
    """

    def __init__(self, client: EKSPodClient) -> None:
        """Initialize a `PodWatcher` instance."""
        self._client = client
        self._entries: dict[str, _PodEntry] = {}
        self._lock = asyncio.Lock()
        self._task: asyncio.Task[None] | None = None
        self._stopped = False

    async def start(self) -> None:
        """Spawn the watch loop. Idempotent."""
        if self._task is None:
            self._task = asyncio.create_task(self._loop(), name="PodWatcher")

    async def stop(self) -> None:
        """Cancel the loop and reject any pending futures. Idempotent."""
        self._stopped = True
        if self._task is not None:
            self._task.cancel()
            with suppress(asyncio.CancelledError, Exception):
                await self._task
            self._task = None
        for entry in self._entries.values():
            for fut in (entry.ready_future, entry.dead_future):
                if not fut.done():
                    fut.cancel()
        self._entries.clear()

    async def register(self, pod_name: str) -> _PodEntry:
        """Allocate ready/dead futures for `pod_name`. Must be called BEFORE pod create."""
        async with self._lock:
            entry = self._entries.get(pod_name)
            if entry is None:
                entry = _PodEntry(pod_name, asyncio.get_running_loop())
                self._entries[pod_name] = entry
            return entry

    def unregister(self, pod_name: str) -> None:
        """Drop the entry, cancelling unresolved futures."""
        entry = self._entries.pop(pod_name, None)
        if entry is None:
            return
        for fut in (entry.ready_future, entry.dead_future):
            if not fut.done():
                fut.cancel()

    async def _loop(self) -> None:
        """Run the Watch until `stop()`. Reconnects on 410 Gone, network drops, token rotation."""
        from kubernetes_asyncio.client.exceptions import ApiException
        from kubernetes_asyncio.watch import Watch

        backoff = 1.0
        while not self._stopped:
            try:
                await self._client._ensure_token_fresh()
                assert self._client._core_v1 is not None  # noqa: S101
                async with Watch() as w:
                    stream = w.stream(
                        self._client._core_v1.list_namespaced_pod,
                        namespace=self._client.namespace,
                        label_selector="strands-env/managed=true",
                        timeout_seconds=60,
                    )
                    async for event in stream:
                        if self._stopped:
                            return  # type: ignore[unreachable]
                        self._handle_event(event)
                backoff = 1.0
            except asyncio.CancelledError:
                return
            except ApiException as e:
                if e.status == 410:
                    continue  # resource version expired — re-list
                logger.warning("PodWatcher: ApiException %s; reconnecting in %.1fs", e.status, backoff)
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, 30.0)
            except Exception as e:  # noqa: BLE001 — watch loop must survive everything
                logger.warning("PodWatcher: %s: %s; reconnecting in %.1fs", type(e).__name__, e, backoff)
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, 30.0)

    def _handle_event(self, event: dict[str, Any]) -> None:
        pod = event.get("object")
        if pod is None:
            return
        name: str | None = getattr(getattr(pod, "metadata", None), "name", None)
        if not name:
            return
        entry = self._entries.get(name)
        if entry is None:
            return  # event for an unregistered pod

        if event.get("type") == "DELETED":
            if not entry.dead_future.done():
                entry.dead_future.set_result(SandboxNotFound(f"pod={name!r} deleted"))
            return

        outcome = _classify_status(pod, name)
        if isinstance(outcome, _Ready):
            if not entry.ready_future.done():
                entry.ready_future.set_result(None)
        elif isinstance(outcome, _Dead) and not entry.dead_future.done():
            entry.dead_future.set_result(outcome.error)


def _classify_status(pod: Any, name: str) -> _Ready | _Dead | None:
    """Walk a Pod object; return ready/dead transition or None if still pending."""
    status = getattr(pod, "status", None)
    if status is None:
        return None

    for cond in status.conditions or []:
        if cond.type == "PodScheduled" and cond.status == "False":
            return _Dead(PodStartError(f"pod={name!r} unschedulable (reason={cond.reason!r}): {cond.message!r}"))

    cstatuses = status.container_statuses or []
    cs = cstatuses[0] if cstatuses else None

    if cs is not None and cs.state and cs.state.waiting:
        reason = cs.state.waiting.reason or ""
        msg = cs.state.waiting.message or ""
        if reason in _PERMANENT_WAITING_REASONS or "no space left" in msg.lower():
            return _Dead(PodStartError(f"pod={name!r} waiting permanently: {reason}: {msg!r}"))

    if cs is not None and cs.state and cs.state.terminated:
        t = cs.state.terminated
        # Pod-level reason adds context the container-level reason often hides.
        # In particular `activeDeadlineSeconds` shows up as `pod.status.reason=
        # "DeadlineExceeded"` but the container-level reason is just `"Error"`
        # with exit=137 — indistinguishable from cgroup OOM at the container
        # level alone. Containerd also has a long-running bug where cgroup OOM
        # gets reported as reason="Error" exit=137 (k8s#108441), so exit=137
        # with a non-OOM reason is treated as likely OOM here.
        pod_reason = status.reason or ""
        if t.reason == "OOMKilled" or (t.exit_code == 137 and pod_reason != "DeadlineExceeded"):
            return _Dead(SandboxOOMKilled(
                f"pod={name!r} OOMKilled (container_reason={t.reason!r} pod_reason={pod_reason!r} exit={t.exit_code})"
            ))
        return _Dead(SandboxDied(
            f"pod={name!r} container terminated container_reason={t.reason!r} "
            f"pod_reason={pod_reason!r} exit={t.exit_code} pod_message={status.message!r}"
        ))

    if cs is not None and cs.last_state and cs.last_state.terminated:
        lt = cs.last_state.terminated
        if lt.reason == "OOMKilled":
            return _Dead(SandboxOOMKilled(f"pod={name!r} previously OOMKilled (exit={lt.exit_code})"))

    if status.phase == "Failed":
        reason = status.reason or "Failed"
        cls = SandboxOOMKilled if reason == "OOMKilled" else SandboxDied
        return _Dead(cls(f"pod={name!r} Failed phase reason={reason!r}: {status.message!r}"))
    if status.phase == "Succeeded":
        return _Dead(SandboxDied(f"pod={name!r} entered Succeeded (envd exited?)"))

    if status.phase == "Running" and cs is not None and cs.ready:
        return _Ready()
    return None


# ---------------------------------------------------------------------------
# EKSPodClient
# ---------------------------------------------------------------------------


class EKSPodClient:
    """Process-shared async client for managing pods on an EKS Fargate cluster."""

    def __init__(
        self,
        cluster_name: str,
        region: str = "us-east-1",
        namespace: str = "default",
        *,
        role_arn: str | None = None,
        profile_name: str | None = None,
        ecr_pull_through_cache: str | bool | None = None,
        pod_create_concurrency: int = 50,
        exec_concurrency: int = 200,
        cp_concurrency: int = 50,
        connection_pool_maxsize: int = 2048,
    ) -> None:
        """Initialize an `EKSPodClient`.

        `ecr_pull_through_cache` accepts:
          - `str`  — used verbatim, e.g. `"<acct>.dkr.ecr.<region>.amazonaws.com/docker-hub"`.
          - `True` — auto-compose `<acct>.dkr.ecr.<region>.amazonaws.com/docker-hub` after
                     `connect()` resolves the boto session (account id via STS). Matches
                     harbor-aws's `dockerhub_cache_enabled=True` convention; the upstream
                     ECR pull-through rule with prefix `docker-hub` must already exist.
          - `None` / `False` — no rewriting; images pulled from upstream directly.
        """
        self.cluster_name, self.region, self.namespace = cluster_name, region, namespace
        self.role_arn, self.profile_name = role_arn, profile_name
        # Resolved to a string in `connect()` when caller passes `True`.
        self.ecr_pull_through_cache: str | bool | None = ecr_pull_through_cache
        self.pod_create_concurrency = pod_create_concurrency
        self.exec_concurrency = exec_concurrency
        self.cp_concurrency = cp_concurrency
        self.connection_pool_maxsize = connection_pool_maxsize

        self._api_client: ApiClient | None = None
        self._api_config: Configuration | None = None
        self._core_v1: CoreV1Api | None = None
        self._endpoint = ""
        self._token = ""
        self._token_expiry: datetime = datetime.now(UTC)
        self._token_lock = asyncio.Lock()
        self._connect_lock = asyncio.Lock()
        self._closed = False
        self._envd_configmap_name = ""
        self._create_sem: asyncio.Semaphore | None = None
        self._exec_sem: asyncio.Semaphore | None = None
        self._cp_sem: asyncio.Semaphore | None = None
        self._watcher: PodWatcher | None = None

    # --- lifecycle ---------------------------------------------------------

    async def connect(self) -> None:
        """Resolve endpoint, mint first token, open API client, start watcher."""
        if self._api_client is not None:
            return
        async with self._connect_lock:
            if self._api_client is not None:  # type: ignore[unreachable]
                return  # type: ignore[unreachable]

            import aiohttp
            from kubernetes_asyncio.client import ApiClient, Configuration, CoreV1Api
            from kubernetes_asyncio.config import load_kube_config_from_dict

            from strands_env.utils.aws import get_session

            boto_session = get_session(
                region_name=self.region, profile_name=self.profile_name, role_arn=self.role_arn
            )
            eks = boto_session.client("eks", region_name=self.region)
            cluster = eks.describe_cluster(name=self.cluster_name)["cluster"]
            self._endpoint = cluster["endpoint"]
            ca_data_b64 = cluster["certificateAuthority"]["data"]
            self._token, self._token_expiry = generate_eks_token(self.cluster_name, boto_session)
            self._boto_session = boto_session

            # Resolve `ecr_pull_through_cache=True` → "<account>.dkr.ecr.<region>.amazonaws.com/docker-hub".
            # Matches harbor-aws's `dockerhub_cache_enabled` convention: same fixed `docker-hub`
            # namespace, account discovered from STS via the (post-assume-role) boto session.
            if self.ecr_pull_through_cache is True:
                account_id = boto_session.client("sts").get_caller_identity()["Account"]
                self.ecr_pull_through_cache = (
                    f"{account_id}.dkr.ecr.{self.region}.amazonaws.com/docker-hub"
                )
                logger.info("ECR pull-through cache resolved: %s", self.ecr_pull_through_cache)
            elif self.ecr_pull_through_cache is False:
                self.ecr_pull_through_cache = None

            config = Configuration()
            await load_kube_config_from_dict(
                config_dict={
                    "apiVersion": "v1",
                    "kind": "Config",
                    "clusters": [
                        {
                            "name": "c",
                            "cluster": {"server": self._endpoint, "certificate-authority-data": ca_data_b64},
                        }
                    ],
                    "users": [{"name": "u", "user": {"token": self._token}}],
                    "contexts": [{"name": "ctx", "context": {"cluster": "c", "user": "u"}}],
                    "current-context": "ctx",
                },
                client_configuration=config,
            )
            config.connection_pool_maxsize = self.connection_pool_maxsize
            self._api_config = config
            api_client = ApiClient(configuration=config)
            # Swap in a connector with SO_KEEPALIVE so dead peers are detected
            # ahead of LB / VPC endpoint 350s resets. Best-effort: skip silently
            # if the internal layout changes upstream.
            with suppress(Exception):
                pool_mgr = api_client.rest_client.pool_manager
                old = pool_mgr.connector
                new = aiohttp.TCPConnector(
                    limit=self.connection_pool_maxsize,
                    socket_factory=_make_keepalive_socket,
                    ssl=getattr(old, "_ssl", None),  # type: ignore[arg-type]
                )
                pool_mgr._connector = new
                if old is not None:
                    await old.close()
            self._core_v1 = CoreV1Api(api_client)
            self._api_client = api_client
            logger.info(
                "EKSPodClient connected: cluster=%s endpoint=%s namespace=%s",
                self.cluster_name,
                self._endpoint,
                self.namespace,
            )

            await self._ensure_envd_configmap()
            self._watcher = PodWatcher(self)
            await self._watcher.start()

    async def close(self) -> None:
        """Stop watcher, close API client. Idempotent."""
        if self._closed:
            return
        self._closed = True
        if self._watcher is not None:
            with suppress(Exception):
                await self._watcher.stop()
            self._watcher = None
        if self._api_client is not None:
            with suppress(Exception):
                await self._api_client.close()
            self._api_client = None
            self._core_v1 = None
        logger.info("EKSPodClient closed: cluster=%s", self.cluster_name)

    async def __aenter__(self) -> EKSPodClient:
        """Connect and return self."""
        await self.connect()
        return self

    async def __aexit__(self, *exc: object) -> None:
        """Close the client."""
        await self.close()

    async def _ensure_token_fresh(self) -> None:
        """Lazy-refresh the EKS bearer token within 60s of expiry."""
        if self._token_expiry - datetime.now(UTC) > timedelta(seconds=60):
            return
        async with self._token_lock:
            if self._token_expiry - datetime.now(UTC) > timedelta(seconds=60):
                return
            token, expiry = await asyncio.to_thread(generate_eks_token, self.cluster_name, self._boto_session)
            self._token = token
            self._token_expiry = expiry
            if self._api_config is not None:
                # `load_kube_config_from_dict` stores the full "Bearer <token>"
                # in api_key["BearerToken"]; we must include the prefix here too.
                self._api_config.api_key["BearerToken"] = f"Bearer {token}"

    def _ensure_semaphores(self) -> None:
        """Create concurrency semaphores on first use (needs a running event loop)."""
        if self._create_sem is None:
            self._create_sem = asyncio.Semaphore(self.pod_create_concurrency)
        if self._exec_sem is None:
            self._exec_sem = asyncio.Semaphore(self.exec_concurrency)
        if self._cp_sem is None:
            self._cp_sem = asyncio.Semaphore(self.cp_concurrency)

    # --- envd ConfigMap ----------------------------------------------------

    async def _ensure_envd_configmap(self) -> None:
        """Create the content-hashed envd ConfigMap. Idempotent across processes (409 is OK)."""
        from kubernetes_asyncio.client.exceptions import ApiException

        assert self._core_v1 is not None  # noqa: S101
        await self._ensure_token_fresh()
        name = f"strands-envd-{_ENVD_SCRIPT_HASH}"
        body = {
            "apiVersion": "v1",
            "kind": "ConfigMap",
            "metadata": {
                "name": name,
                "namespace": self.namespace,
                "labels": {"strands-env/managed": "true", "strands-env/component": "envd"},
            },
            "data": {"envd.py": _ENVD_SCRIPT_CONTENT},
        }
        try:
            await self._core_v1.create_namespaced_config_map(namespace=self.namespace, body=body)  # type: ignore[arg-type]
            logger.info("envd ConfigMap created: %s/%s", self.namespace, name)
        except ApiException as e:
            if e.status != 409:
                raise
        self._envd_configmap_name = name

    # --- pod lifecycle -----------------------------------------------------

    @_retry
    async def _api_create_pod(self, body: dict[str, Any]) -> Any:
        """Create a pod with retry on transient apiserver errors. 409 = idempotent success."""
        from kubernetes_asyncio.client.exceptions import ApiException

        assert self._core_v1 is not None  # noqa: S101
        await self._ensure_token_fresh()
        try:
            return await self._core_v1.create_namespaced_pod(namespace=self.namespace, body=body)  # type: ignore[arg-type]
        except ApiException as e:
            if e.status == 409:
                return None
            raise

    @_retry
    async def _delete_pod(self, pod_name: str, *, grace_period_sec: int = 0) -> None:
        """Delete a pod. Idempotent — 404 is success, transient 5xx auto-retried."""
        from kubernetes_asyncio.client.exceptions import ApiException

        if self._core_v1 is None:
            return
        await self._ensure_token_fresh()
        try:
            await self._core_v1.delete_namespaced_pod(
                name=pod_name, namespace=self.namespace, grace_period_seconds=grace_period_sec
            )
        except ApiException as e:
            if e.status == 404:
                return
            raise

    async def start_pod(
        self,
        image: str,
        *,
        env: dict[str, str] | None = None,
        cpu: str = "1",
        memory: str = "2Gi",
        storage: str = "50Gi",
        labels: dict[str, str] | None = None,
        pull_policy: str = "IfNotPresent",
        startup_timeout_sec: float = 300.0,
        name_prefix: str = "strands",
        service_account: str | None = None,
        node_selector: dict[str, str] | None = None,
        active_deadline_sec: int = DEFAULT_POD_ACTIVE_DEADLINE_SEC,
    ) -> EKSPod:
        """Create a pod, wait for `Ready` via the shared watcher, return a handle.

        The container's `command` is overridden to run envd as PID 1 — the user
        image's ENTRYPOINT is bypassed by design.

        Args:
            image: Container image. Rewritten through `ecr_pull_through_cache` if set.
            env: Environment variables for the container.
            cpu: CPU request (also used as limit on Fargate).
            memory: Memory request (also used as limit on Fargate).
            storage: Ephemeral storage. Default 50Gi — Fargate's 20Gi default includes
                image layers; ML images (PyTorch/CUDA) need more headroom.
            labels: Extra pod labels (we always add `strands-env/managed=true`).
            pull_policy: `Always` / `IfNotPresent` / `Never`.
            startup_timeout_sec: Max time to wait for Ready.
            name_prefix: Prefix for the generated pod name.
            service_account: Optional ServiceAccount (for IRSA).
            node_selector: Optional node selector.
            active_deadline_sec: Hard wall-clock cap on pod lifetime; last-resort leak guard.

        Returns:
            An `EKSPod` handle bound to this client and the watcher entry.
        """
        await self.connect()
        self._ensure_semaphores()
        assert self._core_v1 is not None and self._create_sem is not None  # noqa: S101
        assert self._watcher is not None  # noqa: S101

        pod_name = f"{name_prefix}-{uuid.uuid4().hex[:12]}"
        # By this point `connect()` has resolved any bool to a string (or None).
        cache_prefix = self.ecr_pull_through_cache if isinstance(self.ecr_pull_through_cache, str) else None
        resolved_image = resolve_image(image, cache_prefix)
        spec = self._build_pod_spec(
            pod_name=pod_name,
            image=resolved_image,
            env=env,
            cpu=cpu,
            memory=memory,
            storage=storage,
            labels=labels,
            pull_policy=pull_policy,
            service_account=service_account,
            node_selector=node_selector,
            active_deadline_sec=active_deadline_sec,
        )

        # Register BEFORE create — avoids a race where the watcher sees ready
        # events before we have an entry to fire.
        entry = await self._watcher.register(pod_name)
        t_start = time.monotonic()
        logger.debug("start_pod: pod=%s image=%s cpu=%s memory=%s", pod_name, resolved_image, cpu, memory)
        try:
            async with self._create_sem:
                await self._api_create_pod(spec)
            await self._wait_ready_or_dead(entry, startup_timeout_sec, pod_name, resolved_image)
        except BaseException as e:
            logger.warning(
                "start_pod failed: pod=%s elapsed=%.1fs err=%s: %s (cleaning up)",
                pod_name,
                time.monotonic() - t_start,
                type(e).__name__,
                e,
            )
            await self._delete_pod(pod_name, grace_period_sec=0)
            self._watcher.unregister(pod_name)
            raise
        logger.debug("start_pod ready: pod=%s elapsed=%.1fs", pod_name, time.monotonic() - t_start)
        return EKSPod(client=self, name=pod_name, namespace=self.namespace, image=resolved_image, entry=entry)

    async def _wait_ready_or_dead(self, entry: _PodEntry, timeout_sec: float, pod_name: str, image: str) -> None:
        """Block on the entry's ready/dead future, or timeout."""
        done, _pending = await asyncio.wait(
            [entry.ready_future, entry.dead_future],  # type: ignore[type-var]
            return_when=asyncio.FIRST_COMPLETED,
            timeout=timeout_sec,
        )
        if not done:
            raise PodStartError(f"pod={pod_name!r} image={image!r} not ready within {timeout_sec:.0f}s")
        if entry.dead_future in done:
            err = entry.dead_future.result()
            raise PodStartError(str(err)) from err

    def _build_pod_spec(
        self,
        *,
        pod_name: str,
        image: str,
        env: dict[str, str] | None,
        cpu: str,
        memory: str,
        storage: str,
        labels: dict[str, str] | None,
        pull_policy: str,
        service_account: str | None,
        node_selector: dict[str, str] | None,
        active_deadline_sec: int,
    ) -> dict[str, Any]:
        """Build a pod spec dict that runs envd as PID 1."""
        merged_labels = {"strands-env/managed": "true", **(labels or {})}
        container: dict[str, Any] = {
            "name": "main",
            "image": image,
            "imagePullPolicy": pull_policy,
            "command": ["sh", "-c", _ENVD_BOOTSTRAP],
            "resources": {
                "requests": {"cpu": cpu, "memory": memory, "ephemeral-storage": storage},
                "limits": {"cpu": cpu, "memory": memory, "ephemeral-storage": storage},
            },
            "ports": [{"containerPort": ENVD_PORT, "protocol": "TCP"}],
            "volumeMounts": [{"name": "envd-script", "mountPath": _ENVD_MOUNT_PATH, "readOnly": True}],
            # K8s would flag Ready as soon as the container starts, but envd's
            # listener takes 100-500ms to bind. Probing /health removes the race.
            "readinessProbe": {
                "httpGet": {"path": "/health", "port": ENVD_PORT},
                "periodSeconds": 1,
                "timeoutSeconds": 2,
                "failureThreshold": 60,
            },
        }
        if env:
            container["env"] = [{"name": k, "value": v} for k, v in env.items()]
        spec: dict[str, Any] = {
            "containers": [container],
            "restartPolicy": "Never",
            "activeDeadlineSeconds": active_deadline_sec,
            "volumes": [
                {
                    "name": "envd-script",
                    "configMap": {"name": self._envd_configmap_name, "defaultMode": 0o555},
                }
            ],
        }
        if service_account:
            spec["serviceAccountName"] = service_account
        if node_selector:
            spec["nodeSelector"] = node_selector
        return {
            "apiVersion": "v1",
            "kind": "Pod",
            "metadata": {"name": pod_name, "namespace": self.namespace, "labels": merged_labels},
            "spec": spec,
        }


# ---------------------------------------------------------------------------
# EKSPod handle
# ---------------------------------------------------------------------------


class EKSPod:
    """Handle to a single running pod. One-shot: stop after use."""

    def __init__(
        self,
        *,
        client: EKSPodClient,
        name: str,
        namespace: str,
        image: str,
        entry: _PodEntry,
    ) -> None:
        """Initialize an `EKSPod` instance."""
        self.client = client
        self.name = name
        self.namespace = namespace
        self.image = image
        self._entry = entry
        self._stopped = False

    async def __aenter__(self) -> EKSPod:
        """Return self; pod is already started."""
        return self

    async def __aexit__(self, *exc: object) -> None:
        """Best-effort stop on context exit."""
        await self.stop()

    # --- envd RPC plumbing -------------------------------------------------

    def _resource_path(self, path: str) -> str:
        return f"/api/v1/namespaces/{self.namespace}/pods/{self.name}:{ENVD_PORT}/proxy{path}"

    def _check_alive(self) -> None:
        """Raise immediately if the watcher has marked this pod dead."""
        df = self._entry.dead_future
        if df.done() and not df.cancelled():
            raise df.result()

    def _http_error(self, method: str, path: str, status: int, body: bytes) -> EKSPodError:
        if status == 404:
            return SandboxNotFound(f"pod={self.name!r} envd {method} {path}: 404 (pod gone?)")
        return EKSPodError(
            f"pod={self.name!r} envd {method} {path}: HTTP {status}: {body.decode('utf-8', errors='replace')!r}"
        )

    async def _call_envd_json(
        self,
        method: str,
        path: str,
        *,
        body: Any = None,
        content_type: str = "application/json",
        timeout_sec: float | None = None,
        binary_response: bool = False,
        sem: asyncio.Semaphore | None = None,
    ) -> Any:
        """Buffered envd call. Used for non-streaming paths (files, /health, /metrics)."""
        client = self.client
        assert client._api_client is not None  # noqa: S101
        await client._ensure_token_fresh()
        sem = sem or client._exec_sem
        assert sem is not None  # noqa: S101

        headers: dict[str, str] = {"Accept": "application/json"}
        if body is not None:
            headers["Content-Type"] = content_type
        request_timeout = (timeout_sec + 30) if timeout_sec else None
        resource_path = self._resource_path(path)

        self._check_alive()
        async with sem:
            if binary_response:
                resp = await client._api_client.call_api(
                    resource_path=resource_path,
                    method=method,
                    header_params=headers,
                    body=body,
                    auth_settings=["BearerToken"],
                    _preload_content=False,
                    _request_timeout=request_timeout,
                )
                try:
                    data = await resp.read()
                    if resp.status >= 400:
                        raise self._http_error(method, path, resp.status, data[:500])
                    return data
                finally:
                    resp.release()
            return await client._api_client.call_api(
                resource_path=resource_path,
                method=method,
                header_params=headers,
                body=body,
                auth_settings=["BearerToken"],
                response_types_map={200: "object"},
                _return_http_data_only=True,
                _preload_content=True,
                _request_timeout=request_timeout,
            )

    async def metrics(self) -> dict[str, Any]:
        """Pod-level metrics (cgroup memory, load avg) from envd's `/metrics`."""
        return await self._call_envd_json("GET", "/metrics", timeout_sec=5)

    # --- exec (streaming NDJSON) -------------------------------------------

    async def exec(self, command: str | list[str], *, timeout_sec: float | None = None) -> ExecResult:
        """Run a command inside the pod and return the result.

        Wire protocol: NDJSON frames over chunked HTTP. envd sends:

            {"type":"started","pid":N,"t":0.01}
            {"type":"heartbeat","t":30.0}            # every HEARTBEAT_INTERVAL_SEC
            {"type":"result","exit_code":0,"stdout":"...","stderr":"...",
             "truncated":bool,"timeout":bool}

        Client raises `SandboxUnresponsive` if no frame arrives for
        `HEARTBEAT_GRACE_SEC`. The watcher's `dead_future` is cross-checked via
        `asyncio.wait` — pod death cancels the in-flight call immediately.

        **No auto-retry.** A connection error is ambiguous (request may or may
        not have reached envd); retrying would double-run non-idempotent
        commands. Callers that know their command is idempotent can retry at
        their layer.

        Raises:
            PodExecTimeoutError: subprocess exceeded `timeout_sec` budget.
            SandboxUnresponsive: envd stopped emitting frames.
            SandboxDied / SandboxOOMKilled / SandboxNotFound: pod died mid-exec.
            EKSPodError: HTTP transport / envd error.
        """
        import shlex

        cmd_str = command if isinstance(command, str) else shlex.join(command)
        # Pass `body` as a dict — `call_api` JSON-serializes it. Passing
        # pre-encoded bytes makes it try to re-serialize and TypeErrors on bytes.
        body = {"cmd": cmd_str, "timeout_sec": int(timeout_sec) if timeout_sec else None}

        client = self.client
        assert client._api_client is not None and client._exec_sem is not None  # noqa: S101
        await client._ensure_token_fresh()
        self._check_alive()

        t_start = time.monotonic()
        async with client._exec_sem:
            exec_task = asyncio.create_task(self._run_exec_stream(body))
            death_future = self._entry.dead_future
            try:
                done, _pending = await asyncio.wait(
                    [exec_task, death_future],  # type: ignore[type-var]
                    return_when=asyncio.FIRST_COMPLETED,
                )
            except BaseException:
                exec_task.cancel()
                raise
            if death_future in done and exec_task not in done:
                exec_task.cancel()
                with suppress(BaseException):
                    await exec_task
                raise death_future.result()
            result = await exec_task

        logger.debug(
            "exec: pod=%s rc=%d dur=%.1fs cmd=%r",
            self.name,
            result.return_code,
            time.monotonic() - t_start,
            cmd_str[:80],
        )
        return result

    async def _run_exec_stream(self, body: dict[str, Any]) -> ExecResult:
        """POST /commands/run with `_preload_content=False`; parse NDJSON frames."""
        from kubernetes_asyncio.client.exceptions import ApiException

        client = self.client
        assert client._api_client is not None  # noqa: S101
        headers = {"Accept": "application/x-ndjson", "Content-Type": "application/json"}
        try:
            resp = await client._api_client.call_api(
                resource_path=self._resource_path("/commands/run"),
                method="POST",
                header_params=headers,
                body=body,
                auth_settings=["BearerToken"],
                _preload_content=False,
                _request_timeout=DEFAULT_REQUEST_TIMEOUT_SEC,
            )
        except ApiException as e:
            if e.status == 404:
                raise SandboxNotFound(f"pod={self.name!r} exec: 404 (pod gone?)") from e
            raise EKSPodError(f"pod={self.name!r} exec failed: HTTP {e.status} body={(e.body or b'')[:500]!r}") from e

        try:
            if resp.status >= 400:
                data = await resp.read()
                raise self._http_error("POST", "/commands/run", resp.status, data[:500])
            return await self._parse_ndjson_frames(resp)
        finally:
            with suppress(Exception):
                resp.release()

    async def _parse_ndjson_frames(self, resp: Any) -> ExecResult:
        """Read newline-delimited JSON; enforce heartbeat grace; return ExecResult."""
        buf = bytearray()
        seen_started = False

        while True:
            try:
                chunk = await asyncio.wait_for(resp.content.readany(), timeout=HEARTBEAT_GRACE_SEC)
            except asyncio.TimeoutError as e:
                self._check_alive()
                raise SandboxUnresponsive(
                    f"pod={self.name!r} no envd frame for {HEARTBEAT_GRACE_SEC:.0f}s (started={seen_started})"
                ) from e

            if not chunk:
                self._check_alive()
                raise SandboxUnresponsive(
                    f"pod={self.name!r} envd stream closed before result frame (started={seen_started})"
                )

            buf.extend(chunk)
            while True:
                nl = buf.find(b"\n")
                if nl < 0:
                    break
                line = bytes(buf[:nl])
                del buf[: nl + 1]
                if not line:
                    continue
                try:
                    frame = json.loads(line)
                except json.JSONDecodeError:
                    logger.warning("pod=%s envd: undecodable frame: %r", self.name, line[:200])
                    continue
                ftype = frame.get("type")
                if ftype == "started":
                    seen_started = True
                elif ftype == "heartbeat":
                    pass
                elif ftype == "result":
                    if frame.get("timeout"):
                        raise PodExecTimeoutError(
                            f"pod={self.name!r} exec timed out: {frame.get('stderr', '')[:200]!r}"
                        )
                    return ExecResult(
                        stdout=frame.get("stdout", ""),
                        stderr=frame.get("stderr", ""),
                        return_code=int(frame.get("exit_code", -1)),
                        truncated=bool(frame.get("truncated", False)),
                    )
                elif ftype == "error":
                    raise EKSPodError(f"pod={self.name!r} envd error: {frame.get('message', '?')!r}")

    # --- file transfer (tar via envd /files) -------------------------------

    @_retry
    async def upload_file(self, source: str | Path, target: str, *, timeout_sec: float | None = None) -> None:
        """Copy a local file to `target` inside the pod (parent dirs created)."""
        source = Path(source)
        if not source.is_file():
            raise EKSPodError(f"upload_file: source {source!s} is not a file")
        target_dir = str(Path(target).parent or "/")
        buf = io.BytesIO()
        with tarfile.open(fileobj=buf, mode="w") as tar:
            tar.add(str(source), arcname=Path(target).name)
        await self._call_envd_json(
            "PUT",
            f"/files?path={target_dir}",
            body=buf.getvalue(),
            content_type="application/x-tar",
            timeout_sec=timeout_sec,
            sem=self.client._cp_sem,
        )

    @_retry
    async def upload_dir(self, source: str | Path, target: str, *, timeout_sec: float | None = None) -> None:
        """Recursively copy a local directory to `target` inside the pod."""
        source = Path(source)
        if not source.is_dir():
            raise EKSPodError(f"upload_dir: source {source!s} is not a directory")
        buf = io.BytesIO()
        with tarfile.open(fileobj=buf, mode="w") as tar:
            tar.add(str(source), arcname=".")
        await self._call_envd_json(
            "PUT",
            f"/files?path={target}",
            body=buf.getvalue(),
            content_type="application/x-tar",
            timeout_sec=timeout_sec,
            sem=self.client._cp_sem,
        )

    @_retry
    async def download_file(self, source: str, target: str | Path, *, timeout_sec: float | None = None) -> None:
        """Copy a file at `source` inside the pod to local `target`."""
        target = Path(target)
        target.parent.mkdir(parents=True, exist_ok=True)
        tar_bytes = await self._call_envd_json(
            "GET",
            f"/files?path={source}",
            timeout_sec=timeout_sec,
            binary_response=True,
            sem=self.client._cp_sem,
        )
        with tarfile.open(fileobj=io.BytesIO(tar_bytes), mode="r") as tar:
            members = [m for m in tar.getmembers() if m.isfile()]
            if not members:
                raise EKSPodError(f"pod={self.name!r} download_file {source!r}: no file in tar")
            f = tar.extractfile(members[0])
            if f is None:
                raise EKSPodError(f"pod={self.name!r} download_file {source!r}: tar member not extractable")
            target.write_bytes(f.read())

    @_retry
    async def download_dir(self, source: str, target: str | Path, *, timeout_sec: float | None = None) -> None:
        """Recursively copy a directory at `source` inside the pod to local `target`."""
        target = Path(target)
        target.mkdir(parents=True, exist_ok=True)
        tar_bytes = await self._call_envd_json(
            "GET",
            f"/files?path={source}",
            timeout_sec=timeout_sec,
            binary_response=True,
            sem=self.client._cp_sem,
        )
        with tarfile.open(fileobj=io.BytesIO(tar_bytes), mode="r") as tar:
            top = Path(source).name
            for member in tar.getmembers():
                name = member.name
                if name == top:
                    continue
                if name.startswith(f"{top}/"):
                    member.name = name[len(top) + 1 :]
                    if member.name:
                        tar.extract(member, path=target, filter="data")
                else:
                    tar.extract(member, path=target, filter="data")

    # --- misc --------------------------------------------------------------

    @_retry
    async def logs(self, *, tail_lines: int | None = None) -> str:
        """Return container logs (decoded UTF-8). Uses K8s API, not envd."""
        assert self.client._core_v1 is not None  # noqa: S101
        await self.client._ensure_token_fresh()
        kwargs: dict[str, Any] = {"name": self.name, "namespace": self.namespace, "container": "main"}
        if tail_lines is not None:
            kwargs["tail_lines"] = tail_lines
        result = await self.client._core_v1.read_namespaced_pod_log(**kwargs)
        return result if isinstance(result, str) else str(result)

    async def stop(self, *, grace_period_sec: int = 0) -> None:
        """Delete the pod and unregister from the watcher. Idempotent."""
        if self._stopped:
            return
        self._stopped = True
        try:
            await self.client._delete_pod(self.name, grace_period_sec=grace_period_sec)
        finally:
            if self.client._watcher is not None:
                self.client._watcher.unregister(self.name)
