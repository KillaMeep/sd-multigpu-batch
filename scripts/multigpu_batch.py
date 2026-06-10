import os
import sys
import threading
import time

import gradio as gr
import requests

# Add extension root to path so `multigpu_lib.*` imports resolve.
_ext_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ext_root not in sys.path:
    sys.path.insert(0, _ext_root)

import modules.processing as _processing_module
import modules.scripts as scripts

from multigpu_lib.dispatcher import dispatch_all
from multigpu_lib.log import err, info, ok, warn
from multigpu_lib.result_merger import merge_results
from multigpu_lib.seed_utils import resolve_seed, split_seeds_from_work, split_work
from multigpu_lib.worker_manager import (
    _workers,
    get_or_start_worker,
    shutdown_all,
    sync_model_to_worker,
)

# ── Save original process_images before patching ─────────────────────────────
_original_process_images = _processing_module.process_images

# ── Module-level state (updated by Gradio callbacks) ────────────────────────
_enabled: bool = False
_worker_gpu_ids: list = []


# ── Helpers ──────────────────────────────────────────────────────────────────

def _cuda_device_count() -> int:
    try:
        import torch
        return torch.cuda.device_count()
    except Exception:
        return 1


def _gpu_status_html() -> str:
    """Build a coloured per-GPU status panel."""
    n = _cuda_device_count()

    def row(gpu_id, role, color, label):
        return (
            f'<div style="display:flex;align-items:center;gap:8px;padding:3px 0">'
            f'<span style="color:{color};font-size:9px;line-height:1">&#9679;</span>'
            f'<span style="color:#d1d5db">GPU {gpu_id}</span>'
            f'<span style="color:#6b7280;font-size:11px">{role}</span>'
            f'<span style="color:{color};font-size:11px;margin-left:auto">{label}</span>'
            f'</div>'
        )

    rows = [row(0, "primary", "#22c55e", "always ready")]

    for gpu_id in range(1, n):
        w = _workers.get(gpu_id)
        if w is None:
            color, label = "#6b7280", "not started"
        elif w["process"].poll() is not None:
            color, label = "#ef4444", "crashed"
        elif w["ready"]:
            color, label = "#22c55e", f"ready  —  port {w['port']}"
        else:
            color, label = "#f59e0b", "starting…"
        rows.append(row(gpu_id, "worker", color, label))

    inner = "".join(rows)
    return (
        '<div style="font-family:ui-monospace,SFMono-Regular,monospace;font-size:12px;'
        'padding:8px 12px;background:rgba(0,0,0,0.25);border:1px solid rgba(255,255,255,0.08);'
        'border-radius:6px;line-height:1.8">'
        + inner
        + "</div>"
    )


# ── Progress aggregation ──────────────────────────────────────────────────────

def _aggregate_progress(worker_ports: list, total_n_iter: int, stop_event):
    """
    Background thread: watches for the new generation to start, then
    rescales shared.state.job_count to total_n_iter and tracks remote
    job completions so the progress bar reflects all GPUs.
    """
    try:
        from modules import shared
    except Exception:
        return

    remote_ports = [p for p in worker_ports if p is not None]
    last_remote_job_nos = {port: 0 for port in remote_ports}

    # Detect generation start via the transient job_count == -1 set by state.begin()
    pre_count = shared.state.job_count
    saw_begin = False
    deadline = time.time() + 10

    while time.time() < deadline and not stop_event.is_set():
        jc = shared.state.job_count
        if jc == -1:
            saw_begin = True
        elif saw_begin and jc > 0:
            shared.state.job_count = total_n_iter
            break
        elif not saw_begin and jc != pre_count and jc > 0:
            # Missed the -1 window; still caught the change
            shared.state.job_count = total_n_iter
            break
        time.sleep(0.02)

    # Poll remote workers and advance job_no as they complete iterations
    while not stop_event.is_set():
        time.sleep(0.5)
        for port in remote_ports:
            try:
                r = requests.get(
                    f"http://127.0.0.1:{port}/sdapi/v1/progress", timeout=1
                )
                if r.ok:
                    remote_job_no = r.json().get("state", {}).get("job_no", 0)
                    delta = remote_job_no - last_remote_job_nos[port]
                    if delta > 0:
                        shared.state.job_no = min(
                            shared.state.job_no + delta,
                            shared.state.job_count,
                        )
                        last_remote_job_nos[port] = remote_job_no
            except Exception:
                pass


# ── Patched process_images ────────────────────────────────────────────────────

def _multigpu_process_images(p, *args, **kwargs):
    if not _enabled or not _worker_gpu_ids:
        return _original_process_images(p, *args, **kwargs)

    all_gpu_ids = sorted(set([0] + _worker_gpu_ids))
    n_gpus = len(all_gpu_ids)

    worker_ports = [None]
    for dev_id in all_gpu_ids[1:]:
        info(f"Ensuring worker GPU {dev_id} is running...")
        w = get_or_start_worker(dev_id)
        if not w["ready"]:
            err(f"Worker GPU {dev_id} not ready — falling back to single GPU.")
            return _original_process_images(p, *args, **kwargs)
        info(f"Syncing model to worker GPU {dev_id} (port {w['port']})...")
        sync_model_to_worker(w["port"])
        worker_ports.append(w["port"])

    base_seed = resolve_seed(p.seed)
    work_splits = split_work(p.n_iter, p.batch_size, n_gpus)
    seeds = split_seeds_from_work(base_seed, work_splits)
    total_n_iter = sum(ni for ni, _ in work_splits if ni > 0)

    total_images = p.n_iter * p.batch_size
    info(f"Distributing {total_images} image(s) across {n_gpus} GPU(s):")
    for gpu_id, (ni, bs), seed, port in zip(all_gpu_ids, work_splits, seeds, worker_ports):
        if ni == 0:
            continue
        location = "local" if port is None else f"port {port}"
        info(f"  GPU {gpu_id} ({location}): {ni} iter(s) × batch {bs}  seed={seed}")

    # Start progress aggregation background thread
    _stop_progress = threading.Event()
    _progress_thread = threading.Thread(
        target=_aggregate_progress,
        args=(worker_ports, total_n_iter, _stop_progress),
        daemon=True,
    )
    _progress_thread.start()

    info("Dispatching — all GPUs running in parallel...")
    results = dispatch_all(
        p, work_splits, worker_ports, seeds, local_fn=_original_process_images
    )

    _stop_progress.set()
    _progress_thread.join(timeout=2)

    if results[0] is None:
        raise RuntimeError("[MULTIGPU] Primary GPU (GPU 0) generation failed.")

    merged = merge_results(results)
    ok(f"Done — merged {len(merged.images)} image(s) from {n_gpus} GPU(s).")
    return merged


# Apply patch
_processing_module.process_images = _multigpu_process_images

ok(
    f"Extension loaded — {_cuda_device_count()} CUDA device(s) detected. "
    "process_images patched."
)


# ── Script class ──────────────────────────────────────────────────────────────

class MultiGPUBatchScript(scripts.Script):

    def title(self):
        return "Multi-GPU Batch"

    def show(self, is_img2img):
        return scripts.AlwaysVisible

    def ui(self, is_img2img):
        n_gpus = _cuda_device_count()
        worker_choices = [f"GPU {i}" for i in range(1, n_gpus)]

        with gr.Accordion("Multi-GPU Batch", open=False):

            with gr.Row():
                with gr.Column(scale=3):
                    enabled = gr.Checkbox(
                        label="Enable Multi-GPU Batch",
                        value=False,
                    )
                with gr.Column(scale=5):
                    if worker_choices:
                        gr.Markdown(
                            "Splits generation across GPUs in parallel. "
                            "GPU 0 always runs locally."
                        )
                    else:
                        gr.HTML(
                            '<p style="color:#f59e0b;margin:6px 0 0 0;font-size:13px">'
                            "No additional CUDA GPUs detected.</p>"
                        )

            selected_gpus = gr.CheckboxGroup(
                choices=worker_choices,
                value=[],
                label="Worker GPUs",
                interactive=bool(worker_choices),
            )

            status_html = gr.HTML(value=_gpu_status_html())

            with gr.Row():
                prewarm_btn = gr.Button("Pre-warm Workers", variant="primary")
                kill_btn = gr.Button("Kill Workers", variant="stop")

        # ── Callbacks ────────────────────────────────────────────────────────

        def on_settings_change(enabled_val, gpu_labels):
            global _enabled, _worker_gpu_ids
            _enabled = bool(enabled_val)
            _worker_gpu_ids = [int(g.split()[-1]) for g in (gpu_labels or [])]

        enabled.change(fn=on_settings_change, inputs=[enabled, selected_gpus], outputs=[])
        selected_gpus.change(fn=on_settings_change, inputs=[enabled, selected_gpus], outputs=[])

        def prewarm(enabled_val, selected):
            if not enabled_val:
                return _gpu_status_html()
            selected_ids = [int(g.split()[-1]) for g in (selected or [])]
            if not selected_ids:
                return _gpu_status_html()
            for dev_id in selected_ids:
                info(f"Pre-warming worker GPU {dev_id}...")
                get_or_start_worker(dev_id)
            return _gpu_status_html()

        def kill_workers():
            info("Killing all worker processes...")
            shutdown_all()
            ok("All workers stopped.")
            return _gpu_status_html()

        prewarm_btn.click(fn=prewarm, inputs=[enabled, selected_gpus], outputs=[status_html])
        kill_btn.click(fn=kill_workers, inputs=[], outputs=[status_html])

        return [enabled, selected_gpus]
