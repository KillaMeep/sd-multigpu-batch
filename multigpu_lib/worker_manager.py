import atexit
import os
import random
import socket
import subprocess
import sys
import time

import requests

from .log import err, info, ok, warn

SYNC_KEYS = [
    "sd_model_checkpoint",
    "sd_vae",
    "CLIP_stop_at_last_layers",
    "eta_noise_seed_delta",
]

# { device_id: {"process": Popen, "port": int, "ready": bool} }
_workers: dict = {}


def _a1111_root() -> str:
    try:
        import modules.paths as paths
        return paths.script_path
    except Exception:
        return os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))


def find_free_port() -> int:
    """Pick a random available TCP port in the ephemeral range."""
    for _ in range(100):
        port = random.randint(20000, 55000)
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                s.bind(("127.0.0.1", port))
                return port
            except OSError:
                continue
    # Fallback: let the OS pick
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def start_worker(device_id: int, port: int) -> subprocess.Popen:
    root = _a1111_root()
    python = sys.executable
    launch = os.path.join(root, "launch.py")
    cmd = [
        python, launch,
        "--nowebui",
        "--api",
        f"--device-id={device_id}",
        f"--port={port}",
        "--skip-prepare-environment",
        "--skip-install",
    ]
    # Clear COMMANDLINE_ARGS so the worker doesn't inherit the primary's launch flags
    # (e.g. --listen --port 3456 from webui-user.sh would clobber our --port).
    env = os.environ.copy()
    env["COMMANDLINE_ARGS"] = ""

    info(f"Starting worker GPU {device_id} on port {port} ...")
    return subprocess.Popen(
        cmd, cwd=root, env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def wait_for_worker(port: int, timeout: int = 180) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            r = requests.get(f"http://127.0.0.1:{port}/sdapi/v1/memory", timeout=2)
            if r.status_code == 200:
                return True
        except requests.exceptions.ConnectionError:
            pass
        time.sleep(2)
    return False


def is_alive(device_id: int) -> bool:
    w = _workers.get(device_id)
    return w is not None and w["process"].poll() is None


def get_or_start_worker(device_id: int) -> dict:
    """Start worker for device_id if not already running. Returns worker dict."""
    if is_alive(device_id):
        return _workers[device_id]

    port = find_free_port()
    proc = start_worker(device_id, port)
    _workers[device_id] = {"process": proc, "port": port, "ready": False}

    info(f"Waiting for worker GPU {device_id} (port {port}) to become ready...")
    ready = wait_for_worker(port)
    _workers[device_id]["ready"] = ready

    if ready:
        ok(f"Worker GPU {device_id} is ready on port {port}.")
    else:
        err(f"Worker GPU {device_id} failed to start within timeout.")

    return _workers[device_id]


def _primary_api_port() -> int:
    """Return the port the primary A1111 API is listening on."""
    try:
        from modules import shared
        p = getattr(shared.cmd_opts, "port", None)
        return int(p) if p else 7860
    except Exception:
        return 7860


def sync_model_to_worker(port: int, primary_port: int = None):
    """Push relevant model/options from the primary to a worker."""
    if primary_port is None:
        primary_port = _primary_api_port()
    try:
        r = requests.get(f"http://127.0.0.1:{primary_port}/sdapi/v1/options", timeout=10)
        if r.status_code != 200:
            warn(f"Could not read primary options (status {r.status_code})")
            return
        primary_opts = r.json()
        subset = {k: v for k, v in primary_opts.items() if k in SYNC_KEYS and v is not None}
        if subset:
            requests.post(f"http://127.0.0.1:{port}/sdapi/v1/options", json=subset, timeout=60)
    except Exception as e:
        err(f"Failed to sync model to port {port}: {e}")


def get_worker_status() -> dict:
    """Return {device_id: status_string} for all known workers."""
    status = {}
    for dev_id, w in _workers.items():
        if w["process"].poll() is not None:
            status[dev_id] = "dead"
        elif w["ready"]:
            status[dev_id] = f"ready (port {w['port']})"
        else:
            status[dev_id] = "starting"
    return status


def shutdown_all():
    for device_id, w in list(_workers.items()):
        proc = w.get("process")
        if proc and proc.poll() is None:
            info(f"Shutting down worker GPU {device_id}")
            proc.terminate()
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()


atexit.register(shutdown_all)
