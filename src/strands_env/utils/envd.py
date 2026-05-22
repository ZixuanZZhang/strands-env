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

"""In-pod RPC server (`envd`) for `strands_env_eks`.

Runs as PID 1 inside each managed pod. Listens on `0.0.0.0:49983`. Reached from
outside via the Kubernetes API server's pod proxy:

    /api/v1/namespaces/{ns}/pods/{name}:49983/proxy/{path}

Auth: none. The API server enforces RBAC on `pods/proxy`; once a request
arrives at envd, the caller is already authorized.

Endpoints:
    GET  /health                — liveness (returns `{ok: true}`).
    GET  /metrics               — cgroup memory + load avg.
    POST /commands/run          — JSON `{cmd, env?, cwd?, timeout_sec?}`.
                                   Response is **chunked NDJSON**:
                                       {"type":"started","pid":N,"t":0.01}
                                       {"type":"heartbeat","t":30.0}
                                       {"type":"result", ...}
                                   Heartbeats keep the connection alive across
                                   NLB / kubelet idle timers and let the client
                                   detect a wedged subprocess.
    PUT  /files?path=...        — request body is tar bytes; extracted at `path`.
    GET  /files?path=...        — response body is tar of the file or dir at `path`.

stdout/stderr from the subprocess are pumped into a **ring buffer** (10 MiB
each by default). envd's memory is bounded regardless of how chatty the
subprocess is — a runaway logger that writes 1 GB uses 10 MiB of RAM here,
not 1 GB. The ring's `dropped > 0` lights the `truncated` flag in `result`.

stdlib-only on Python 3.10+. Deployed as a ConfigMap-mounted script.
"""

from __future__ import annotations

import json
import os
import shlex
import signal
import subprocess
import sys
import threading
import time
import urllib.parse
from contextlib import suppress
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

PORT = int(os.environ.get("STRANDS_ENVD_PORT", "49983"))

# Hard cap on PUT body size (tar payload).
MAX_BODY_BYTES = 256 * 1024 * 1024  # 256 MiB

# Per-stream ring-buffer cap. Keeps the *tail* — tracebacks are usually at the end.
MAX_STDOUT_BYTES = int(os.environ.get("STRANDS_ENVD_MAX_STDOUT_BYTES", str(10 * 1024 * 1024)))  # 10 MiB

# Fall-back subprocess timeout when the client doesn't pass `timeout_sec`.
DEFAULT_TIMEOUT_SEC = int(os.environ.get("STRANDS_ENVD_DEFAULT_TIMEOUT_SEC", "5400"))  # 90 min

# How often we emit a `heartbeat` frame during a long-running exec.
HEARTBEAT_INTERVAL_SEC = float(os.environ.get("STRANDS_ENVD_HEARTBEAT_INTERVAL_SEC", "30"))


class _RingBuffer:
    """Bounded byte buffer that retains the tail. Thread-safe."""

    __slots__ = ("max", "buf", "dropped", "lock")

    def __init__(self, max_bytes: int) -> None:
        """Initialize a `_RingBuffer` with the given cap."""
        self.max = max_bytes
        self.buf = bytearray()
        self.dropped = 0
        self.lock = threading.Lock()

    def write(self, data: bytes) -> None:
        """Append `data`, evicting from the head if over cap."""
        with self.lock:
            self.buf.extend(data)
            overflow = len(self.buf) - self.max
            if overflow > 0:
                self.dropped += overflow
                del self.buf[:overflow]

    def snapshot(self) -> tuple[bytes, bool]:
        """Return `(current_bytes, truncated)`."""
        with self.lock:
            return bytes(self.buf), self.dropped > 0


def _drain_to_ring(pipe: Any, ring: _RingBuffer) -> None:
    """Pump bytes from `pipe` into `ring` until EOF. Best-effort: swallow any error."""
    try:
        while True:
            chunk = pipe.read(4096)
            if not chunk:
                return
            ring.write(chunk)
    except Exception:  # noqa: BLE001 — drainer must not crash the handler thread
        return


class Handler(BaseHTTPRequestHandler):
    """HTTP handler for envd endpoints. One thread per request (ThreadingHTTPServer)."""

    # Must be HTTP/1.1, NOT 1.0 (the BaseHTTPRequestHandler default). HTTP/1.0
    # has no `Transfer-Encoding: chunked`, so upstream proxies (apiserver pod-proxy)
    # treat our chunked NDJSON body as opaque bytes and pass the framing through to
    # the client unparsed. With 1.1 the hop honors chunked and the client sees
    # clean decoded frames.
    protocol_version = "HTTP/1.1"
    server_version = "strands-envd/0.2"
    sys_version = ""  # don't leak Python version in Server header

    def log_message(self, fmt: str, *args: Any) -> None:
        """Log to stderr (kubelet captures this for `kubectl logs`)."""
        sys.stderr.write(f"[envd] {fmt % args}\n")

    # --- helpers -----------------------------------------------------------

    def _parse(self) -> tuple[str, dict[str, str]]:
        u = urllib.parse.urlparse(self.path)
        return u.path, dict(urllib.parse.parse_qsl(u.query))

    def _read_body(self) -> bytes:
        n = int(self.headers.get("Content-Length", "0"))
        if n > MAX_BODY_BYTES:
            self._send_json(413, {"error": f"body too large: {n} > {MAX_BODY_BYTES}"})
            return b""
        return self.rfile.read(n) if n else b""

    def _send_json(self, status: int, obj: Any) -> None:
        body = json.dumps(obj).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_bytes(self, status: int, content_type: str, body: bytes) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    # --- chunked NDJSON writer --------------------------------------------

    def _begin_ndjson(self) -> None:
        """Send 200 + chunked headers. After this, only `_write_frame` / `_end_chunked`."""
        self.send_response(200)
        self.send_header("Content-Type", "application/x-ndjson")
        self.send_header("Transfer-Encoding", "chunked")
        self.send_header("Cache-Control", "no-store")
        self.end_headers()

    def _write_frame(self, frame: dict[str, Any]) -> None:
        """Write one NDJSON frame as a chunk and flush. Raises OSError if client gone."""
        line = (json.dumps(frame) + "\n").encode("utf-8")
        chunk = f"{len(line):x}\r\n".encode() + line + b"\r\n"
        self.wfile.write(chunk)
        self.wfile.flush()

    def _end_chunked(self) -> None:
        """Write the chunked-encoding terminator. Best-effort."""
        with suppress(OSError):
            self.wfile.write(b"0\r\n\r\n")
            self.wfile.flush()

    # --- routing -----------------------------------------------------------

    def do_GET(self) -> None:  # noqa: N802 — BaseHTTPRequestHandler API
        """Dispatch GET requests."""
        path, qs = self._parse()
        if path == "/health":
            self._send_json(200, {"ok": True})
        elif path == "/metrics":
            self._send_json(200, _read_metrics())
        elif path == "/files":
            target = qs.get("path", "")
            if not target:
                self._send_json(400, {"error": "missing required query param: path"})
                return
            self._download_tar(target)
        else:
            self._send_json(404, {"error": f"unknown path: {path}"})

    def do_POST(self) -> None:  # noqa: N802
        """Dispatch POST requests."""
        path, _ = self._parse()
        if path == "/commands/run":
            self._run_command()
        else:
            self._send_json(404, {"error": f"unknown path: {path}"})

    def do_PUT(self) -> None:  # noqa: N802
        """Dispatch PUT requests."""
        path, qs = self._parse()
        if path == "/files":
            target = qs.get("path", "")
            if not target:
                self._send_json(400, {"error": "missing required query param: path"})
                return
            self._upload_tar(target)
        else:
            self._send_json(404, {"error": f"unknown path: {path}"})

    # --- command execution -------------------------------------------------

    def _run_command(self) -> None:
        """Spawn a subprocess and stream NDJSON frames (started → heartbeat* → result)."""
        body = self._read_body()
        try:
            req = json.loads(body or b"{}")
        except json.JSONDecodeError as e:
            self._send_json(400, {"error": f"invalid JSON: {e}"})
            return
        cmd = req.get("cmd")
        if not isinstance(cmd, str) or not cmd:
            self._send_json(400, {"error": "missing or empty 'cmd'"})
            return
        env = os.environ.copy()
        env.update(req.get("env") or {})
        cwd = req.get("cwd") or None
        timeout_sec = req.get("timeout_sec") or DEFAULT_TIMEOUT_SEC

        # OOM-score routing: user subprocess gets oom_score_adj=100 so the kernel
        # OOM killer prefers it over envd (which stays at 0). `2>/dev/null`
        # swallows EPERM if /proc isn't writable.
        wrapped = f"echo 100 > /proc/self/oom_score_adj 2>/dev/null; exec /bin/sh -c {shlex.quote(cmd)}"
        try:
            proc = subprocess.Popen(  # noqa: S603
                ["/bin/sh", "-c", wrapped],
                env=env,
                cwd=cwd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                # New session so we can SIGKILL the whole process group on
                # client disconnect / timeout. Without this, the subprocess's
                # children orphan and run until the cgroup tears the pod down.
                start_new_session=True,
            )
        except OSError as e:
            self._send_json(500, {"error": f"spawn failed: {e}"})
            return

        # Switch to chunked NDJSON. From here on, any HTTP-level error is signalled
        # via a `result` / `error` frame, not a status code.
        self._begin_ndjson()
        t_start = time.monotonic()
        try:
            self._write_frame({"type": "started", "pid": proc.pid, "t": 0.0})
        except OSError:
            _kill_group(proc)
            return

        stdout_ring = _RingBuffer(MAX_STDOUT_BYTES)
        stderr_ring = _RingBuffer(MAX_STDOUT_BYTES)
        t_out = threading.Thread(target=_drain_to_ring, args=(proc.stdout, stdout_ring), daemon=True)
        t_err = threading.Thread(target=_drain_to_ring, args=(proc.stderr, stderr_ring), daemon=True)
        t_out.start()
        t_err.start()

        next_heartbeat = t_start + HEARTBEAT_INTERVAL_SEC
        deadline = t_start + timeout_sec
        timed_out = False
        try:
            while True:
                now = time.monotonic()
                if now >= deadline:
                    timed_out = True
                    _kill_group(proc)
                    break
                wait_for = max(0.05, min(next_heartbeat - now, deadline - now))
                try:
                    proc.wait(timeout=wait_for)
                    break  # subprocess exited cleanly
                except subprocess.TimeoutExpired:
                    pass
                now = time.monotonic()
                if now >= next_heartbeat:
                    try:
                        self._write_frame({"type": "heartbeat", "t": round(now - t_start, 2)})
                    except OSError:
                        # Client gone — don't leave the subprocess running.
                        _kill_group(proc)
                        return
                    next_heartbeat = now + HEARTBEAT_INTERVAL_SEC
        finally:
            t_out.join(timeout=2)
            t_err.join(timeout=2)

        stdout_bytes, stdout_trunc = stdout_ring.snapshot()
        stderr_bytes, stderr_trunc = stderr_ring.snapshot()
        rc = proc.returncode if proc.returncode is not None else -1
        try:
            self._write_frame(
                {
                    "type": "result",
                    "exit_code": rc,
                    "stdout": stdout_bytes.decode("utf-8", errors="replace"),
                    "stderr": stderr_bytes.decode("utf-8", errors="replace"),
                    "truncated": stdout_trunc or stderr_trunc,
                    "timeout": timed_out,
                }
            )
        except OSError:
            return
        self._end_chunked()

    # --- file transfer (tar streaming) -------------------------------------

    def _upload_tar(self, target: str) -> None:
        """Pipe request body into `tar -xf - -C <target>`. Creates target dir if missing."""
        os.makedirs(target, exist_ok=True)
        body = self._read_body()
        try:
            r = subprocess.run(  # noqa: S603
                ["tar", "-xf", "-", "-C", target],
                input=body,
                capture_output=True,
                check=False,
            )
        except FileNotFoundError:
            self._send_json(500, {"error": "'tar' not found in container PATH"})
            return
        if r.returncode != 0:
            self._send_json(
                500,
                {
                    "error": "tar -xf failed",
                    "exit_code": r.returncode,
                    "stderr": r.stderr.decode("utf-8", errors="replace")[:2000],
                },
            )
            return
        self._send_json(200, {"ok": True, "bytes": len(body)})

    def _download_tar(self, source: str) -> None:
        """Tar up `source` (file or dir) and stream as the response body."""
        if not os.path.exists(source):
            self._send_json(404, {"error": f"path not found: {source}"})
            return
        parent = os.path.dirname(os.path.abspath(source)) or "/"
        name = os.path.basename(os.path.abspath(source))
        try:
            r = subprocess.run(  # noqa: S603
                ["tar", "-cf", "-", "-C", parent, name],
                capture_output=True,
                check=False,
            )
        except FileNotFoundError:
            self._send_json(500, {"error": "'tar' not found in container PATH"})
            return
        if r.returncode != 0:
            self._send_json(
                500,
                {
                    "error": "tar -cf failed",
                    "exit_code": r.returncode,
                    "stderr": r.stderr.decode("utf-8", errors="replace")[:2000],
                },
            )
            return
        self._send_bytes(200, "application/x-tar", r.stdout)


def _kill_group(proc: subprocess.Popen[bytes]) -> None:
    """SIGTERM → SIGKILL the subprocess's process group. Falls back to per-proc kill."""
    try:
        pgid = os.getpgid(proc.pid)
    except OSError:
        pgid = None
    if pgid is None:
        with suppress(OSError):
            proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            with suppress(OSError):
                proc.kill()
        return
    with suppress(OSError):
        os.killpg(pgid, signal.SIGTERM)
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        with suppress(OSError):
            os.killpg(pgid, signal.SIGKILL)


def _read_metrics() -> dict[str, Any]:
    """Read pod-level memory/cpu metrics from cgroup v2 + /proc.

    `/proc/meminfo` would show the *host's* memory inside a container — useless.
    cgroup v2 (`/sys/fs/cgroup/memory.{current,max}`) reports the actual container
    budget. Fall back to cgroup v1 paths for older runtimes; omit fields that
    can't be read (debug data is best-effort).
    """
    m: dict[str, Any] = {}
    for cur_path, max_path in (
        ("/sys/fs/cgroup/memory.current", "/sys/fs/cgroup/memory.max"),
        ("/sys/fs/cgroup/memory/memory.usage_in_bytes", "/sys/fs/cgroup/memory/memory.limit_in_bytes"),
    ):
        try:
            with open(cur_path) as f:
                m["mem_current_bytes"] = int(f.read().strip())
            with open(max_path) as f:
                v = f.read().strip()
                m["mem_max_bytes"] = None if v == "max" else int(v)
            break
        except (FileNotFoundError, ValueError):
            continue
    try:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    m["envd_rss_kb"] = int(line.split()[1])
                    break
    except OSError:
        pass
    try:
        with open("/proc/loadavg") as f:
            m["load_1m"] = float(f.read().split()[0])
    except (OSError, ValueError, IndexError):
        pass
    return m


def main() -> None:
    """Run the envd server forever on `PORT`."""
    server = ThreadingHTTPServer(("0.0.0.0", PORT), Handler)
    sys.stderr.write(f"[envd] listening on 0.0.0.0:{PORT}\n")
    sys.stderr.flush()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
