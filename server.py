#!/usr/bin/env python3
"""
server.py - Launch and manage OpenAI-compatible inference servers.

Supports llama.cpp, LM Studio, and vLLM backends.
Can be used standalone (CLI) or imported as a module by bench.py.

Usage:
    python server.py llama-cpp --model-path /path/to/model.gguf
    python server.py llama-cpp --model-path /path/to/model.gguf --draft-path /path/to/draft.gguf
    python server.py lm-studio
    python server.py vllm --model meta-llama/Llama-3-8B
"""

from __future__ import annotations

import abc
import argparse
import os
import signal
import shutil
import subprocess
import sys
import time

import requests


# ---------------------------------------------------------------------------
# Default paths
# ---------------------------------------------------------------------------

def _find_llama_server() -> str | None:
    """Search for llama-server binary in common locations."""
    import shutil

    # Check if in PATH
    llama_cmd = shutil.which("llama-server")
    if llama_cmd:
        return llama_cmd

    # Common build locations
    candidates = [
        "~/llama.cpp/build/bin/llama-server",
        "~/Code/llama.cpp/build/bin/llama-server",
        "/usr/local/bin/llama-server",
        "/opt/llama.cpp/build/bin/llama-server",
    ]

    for path in candidates:
        expanded = os.path.expanduser(path)
        if os.path.isfile(expanded):
            return expanded

    return None

_DEFAULT_LLAMA_BIN = _find_llama_server()


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------

class ServerBackend(abc.ABC):
    """Abstract base for inference server backends."""

    def __init__(self, host: str = "127.0.0.1", port: int = 8080):
        self.host = host
        self.port = port
        self._process: subprocess.Popen | None = None

    @property
    def base_url(self) -> str:
        return f"http://{self.host}:{self.port}/v1"

    @abc.abstractmethod
    def start(self) -> None:
        """Launch the server process."""

    def stop(self) -> None:
        """Terminate the server process."""
        if self._process and self._process.poll() is None:
            self._process.terminate()
            try:
                self._process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self._process.kill()
                self._process.wait()
            print(f"  Server stopped (pid {self._process.pid})")
            self._process = None
        log_fh = getattr(self, "_log_fh", None)
        if log_fh:
            log_fh.close()
            self._log_fh = None

    def wait_ready(self, timeout: float = 60) -> bool:
        """Poll the server until it responds or timeout is reached."""
        health_url = f"http://{self.host}:{self.port}/health"
        models_url = f"http://{self.host}:{self.port}/v1/models"
        use_health = True
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            try:
                if use_health:
                    # Prefer /health (llama.cpp returns 200 only when model is loaded)
                    r = requests.get(health_url, timeout=2)
                    if r.status_code == 200:
                        return True
                    if r.status_code == 503:
                        # Model still loading, keep polling /health
                        pass
                    else:
                        # /health not supported (404 etc.), fall back to /v1/models
                        use_health = False
                else:
                    r = requests.get(models_url, timeout=2)
                    if r.status_code == 200:
                        return True
            except requests.ConnectionError:
                pass
            except requests.RequestException:
                pass
            # check if the process died
            if self._process and self._process.poll() is not None:
                code = self._process.returncode
                print(f"  Server process exited with code {code}", file=sys.stderr)
                return False
            time.sleep(1)
        return False


# ---------------------------------------------------------------------------
# llama.cpp
# ---------------------------------------------------------------------------

class LlamaCppBackend(ServerBackend):
    """Launches llama-server from llama.cpp."""

    def __init__(
        self,
        model_path: str,
        draft_path: str | None = None,
        host: str = "127.0.0.1",
        port: int = 8080,
        gpu_layers: int = 99,
        ctx_size: int = 4096,
        llama_bin: str | None = None,
        extra_args: list[str] | None = None,
        log_file: str | None = None,
    ):
        super().__init__(host, port)
        self.model_path = model_path
        self.draft_path = draft_path
        self.gpu_layers = gpu_layers
        self.ctx_size = ctx_size
        self.llama_bin = llama_bin or _DEFAULT_LLAMA_BIN
        self.extra_args = extra_args or []
        self.log_file = log_file

    def _build_cmd(self) -> list[str]:
        cmd = [
            self.llama_bin,
            "-m", self.model_path,
            "--host", self.host,
            "--port", str(self.port),
            "-ngl", str(self.gpu_layers),
            "-c", str(self.ctx_size),
        ]
        if self.draft_path:
            cmd += ["--model-draft", self.draft_path]
        cmd += self.extra_args
        return cmd

    def start(self) -> None:
        if not self.llama_bin or not os.path.isfile(self.llama_bin):
            raise FileNotFoundError(f"llama-server binary not found at {self.llama_bin}")
        if not os.path.isfile(self.model_path):
            raise FileNotFoundError(f"Model file not found at {self.model_path}")
        if self.draft_path and not os.path.isfile(self.draft_path):
            raise FileNotFoundError(f"Draft model file not found at {self.draft_path}")

        cmd = self._build_cmd()
        label = "llama.cpp"
        if self.draft_path:
            label += " (speculative)"
        print(f"  [{label}] Starting server on {self.host}:{self.port}")
        print(f"  Command: {' '.join(cmd)}")

        if self.log_file:
            self._log_fh = open(self.log_file, "w")
            out = self._log_fh
        else:
            self._log_fh = None
            out = sys.stderr

        self._process = subprocess.Popen(
            cmd,
            stdout=out,
            stderr=out,
        )
        print(f"  [{label}] Server pid={self._process.pid}")


# ---------------------------------------------------------------------------
# LM Studio
# ---------------------------------------------------------------------------

class LMStudioBackend(ServerBackend):
    """Connects to an existing LM Studio server instance."""

    def __init__(self, host: str = "127.0.0.1", port: int = 1234):
        super().__init__(host, port)

    def start(self) -> None:
        # Try lms CLI first
        lms = shutil.which("lms")
        if lms:
            print(f"  [LM Studio] Found lms CLI at {lms}")
            print(f"  [LM Studio] Starting server via CLI ...")
            subprocess.run([lms, "server", "start"], check=False)
        else:
            print(f"  [LM Studio] No lms CLI found.")
            print(f"  [LM Studio] Make sure LM Studio is running with the server enabled on port {self.port}.")

    def stop(self) -> None:
        # LM Studio lifecycle is managed by the user / GUI
        lms = shutil.which("lms")
        if lms:
            subprocess.run([lms, "server", "stop"], check=False)
        else:
            print(f"  [LM Studio] Server lifecycle managed by LM Studio app.")


# ---------------------------------------------------------------------------
# vLLM
# ---------------------------------------------------------------------------

class VLLMBackend(ServerBackend):
    """Launches vLLM's OpenAI-compatible server."""

    def __init__(
        self,
        model: str,
        draft_model: str | None = None,
        host: str = "0.0.0.0",
        port: int = 8000,
        extra_args: list[str] | None = None,
    ):
        super().__init__(host, port)
        self.model = model
        self.draft_model = draft_model
        self.extra_args = extra_args or []

    def _build_cmd(self) -> list[str]:
        cmd = [
            sys.executable, "-m", "vllm.entrypoints.openai.api_server",
            "--model", self.model,
            "--host", self.host,
            "--port", str(self.port),
        ]
        if self.draft_model:
            cmd += ["--speculative-model", self.draft_model]
        cmd += self.extra_args
        return cmd

    def start(self) -> None:
        cmd = self._build_cmd()
        label = "vLLM"
        if self.draft_model:
            label += " (speculative)"
        print(f"  [{label}] Starting server on {self.host}:{self.port}")
        print(f"  Command: {' '.join(cmd)}")

        self._process = subprocess.Popen(
            cmd,
            stdout=sys.stdout,
            stderr=sys.stderr,
        )
        print(f"  [{label}] Server pid={self._process.pid}")


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def create_backend(backend_type: str, **kwargs) -> ServerBackend:
    """Create a server backend by name.

    Args:
        backend_type: One of "llama-cpp", "lm-studio", "vllm".
        **kwargs: Passed to the backend constructor.
    """
    backends = {
        "llama-cpp": LlamaCppBackend,
        "lm-studio": LMStudioBackend,
        "vllm": VLLMBackend,
    }
    cls = backends.get(backend_type)
    if cls is None:
        raise ValueError(f"Unknown backend: {backend_type!r}. Choose from: {list(backends)}")
    return cls(**kwargs)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _run_server(backend: ServerBackend):
    """Start the server, wait for readiness, then block until Ctrl+C."""
    backend.start()

    print(f"\n  Waiting for server to be ready ...")
    if backend.wait_ready(timeout=120):
        print(f"  Server ready at {backend.base_url}")
        print(f"  Press Ctrl+C to stop.\n")
    else:
        print(f"  Server failed to become ready.", file=sys.stderr)
        backend.stop()
        sys.exit(1)

    # Block until interrupted
    try:
        while True:
            # If the process died, exit
            if backend._process and backend._process.poll() is not None:
                print(f"\n  Server process exited (code {backend._process.returncode})")
                break
            time.sleep(1)
    except KeyboardInterrupt:
        print(f"\n  Shutting down ...")
    finally:
        backend.stop()


def main():
    parser = argparse.ArgumentParser(
        description="Launch OpenAI-compatible inference servers.",
    )
    subparsers = parser.add_subparsers(dest="backend", required=True)

    # -- llama-cpp --
    p_llama = subparsers.add_parser("llama-cpp", help="Launch llama.cpp server")
    p_llama.add_argument("--model-path", required=True, help="Path to the GGUF model file")
    p_llama.add_argument("--draft-path", default=None, help="Path to a draft GGUF model for speculative decoding")
    p_llama.add_argument("--host", default="127.0.0.1", help="Bind address (default: 127.0.0.1)")
    p_llama.add_argument("--port", type=int, default=8080, help="Port (default: 8080)")
    p_llama.add_argument("--gpu-layers", type=int, default=99, help="Number of GPU layers to offload (default: 99)")
    p_llama.add_argument("--ctx-size", type=int, default=4096, help="Context size (default: 4096)")
    p_llama.add_argument("--llama-bin", default=None, help="Path to llama-server binary (auto-detected from PATH or common locations)")
    p_llama.add_argument("extra_args", nargs="*", help="Extra arguments passed to llama-server")

    # -- lm-studio --
    p_lms = subparsers.add_parser("lm-studio", help="Connect to LM Studio server")
    p_lms.add_argument("--host", default="127.0.0.1", help="LM Studio host (default: 127.0.0.1)")
    p_lms.add_argument("--port", type=int, default=1234, help="LM Studio port (default: 1234)")

    # -- vllm --
    p_vllm = subparsers.add_parser("vllm", help="Launch vLLM server")
    p_vllm.add_argument("--model", required=True, help="Model name or path")
    p_vllm.add_argument("--draft-model", default=None, help="Draft model for speculative decoding")
    p_vllm.add_argument("--host", default="0.0.0.0", help="Bind address (default: 0.0.0.0)")
    p_vllm.add_argument("--port", type=int, default=8000, help="Port (default: 8000)")
    p_vllm.add_argument("extra_args", nargs="*", help="Extra arguments passed to vLLM")

    args = parser.parse_args()

    if args.backend == "llama-cpp":
        backend = LlamaCppBackend(
            model_path=args.model_path,
            draft_path=args.draft_path,
            host=args.host,
            port=args.port,
            gpu_layers=args.gpu_layers,
            ctx_size=args.ctx_size,
            llama_bin=args.llama_bin,
            extra_args=args.extra_args,
        )
    elif args.backend == "lm-studio":
        backend = LMStudioBackend(host=args.host, port=args.port)
    elif args.backend == "vllm":
        backend = VLLMBackend(
            model=args.model,
            draft_model=args.draft_model,
            host=args.host,
            port=args.port,
            extra_args=args.extra_args,
        )
    else:
        parser.error(f"Unknown backend: {args.backend}")
        return

    _run_server(backend)


if __name__ == "__main__":
    main()
