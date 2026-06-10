import os
import sys

import gradio as gr

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
    get_or_start_worker,
    shutdown_all,
    sync_model_to_worker,
)

# ── Save original process_images before patching ─────────────────────────────
_original_process_images = _processing_module.process_images
# ─────────────────────────────────────────────────────────────────────────────

# ── Module-level state (updated by Gradio callbacks) ────────────────────────
_enabled: bool = False
_worker_gpu_ids: list = []
# ─────────────────────────────────────────────────────────────────────────────


def _multigpu_process_images(p, *args, **kwargs):
    """
    Drop-in replacement for process_images.
    Dispatches across multiple GPUs when enabled, otherwise falls through.
    """
    if not _enabled or not _worker_gpu_ids:
        return _original_process_images(p, *args, **kwargs)

    all_gpu_ids = sorted(set([0] + _worker_gpu_ids))
    n_gpus = len(all_gpu_ids)

    worker_ports = [None]  # slot 0 = local primary
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

    total_images = p.n_iter * p.batch_size
    info(f"Distributing {total_images} image(s) across {n_gpus} GPU(s):")
    for gpu_id, (ni, bs), seed, port in zip(all_gpu_ids, work_splits, seeds, worker_ports):
        if ni == 0:
            continue
        location = "local" if port is None else f"port {port}"
        info(f"  GPU {gpu_id} ({location}): {ni} iter(s) × batch {bs}  seed={seed}")

    info("Dispatching — all GPUs running in parallel...")
    results = dispatch_all(p, work_splits, worker_ports, seeds, local_fn=_original_process_images)

    if results[0] is None:
        raise RuntimeError("[MULTIGPU] Primary GPU (GPU 0) generation failed.")

    merged = merge_results(results)
    ok(f"Done — merged {len(merged.images)} image(s) from {n_gpus} GPU(s).")
    return merged


# Patch at load time so every generation goes through our function.
_processing_module.process_images = _multigpu_process_images


def _cuda_device_count() -> int:
    try:
        import torch
        return torch.cuda.device_count()
    except Exception:
        return 1


# ── Load banner ───────────────────────────────────────────────────────────────
_n_gpus_detected = _cuda_device_count()
ok(f"Extension loaded — {_n_gpus_detected} CUDA device(s) detected. process_images patched.")
# ─────────────────────────────────────────────────────────────────────────────


class MultiGPUBatchScript(scripts.Script):

    def title(self):
        return "Multi-GPU Batch"

    def show(self, is_img2img):
        # AlwaysVisible: UI panel is always shown; no Script dropdown selection needed.
        return scripts.AlwaysVisible

    def ui(self, is_img2img):
        n_gpus = _cuda_device_count()
        worker_choices = [f"GPU {i}" for i in range(1, n_gpus)]

        with gr.Accordion("Multi-GPU Batch", open=False):
            enabled = gr.Checkbox(label="Enable Multi-GPU splitting", value=False)

            selected_gpus = gr.CheckboxGroup(
                label="Worker GPUs (in addition to GPU 0)",
                choices=worker_choices,
                value=[],
                interactive=bool(worker_choices),
                info="GPU 0 always runs locally. Select additional GPUs to use as workers."
                     if worker_choices else "No additional CUDA GPUs detected.",
            )

            with gr.Row():
                prewarm_btn = gr.Button("Pre-warm Workers")
                kill_btn = gr.Button("Kill Workers", variant="stop")
                status_box = gr.Textbox(
                    label="Worker Status",
                    value="Workers not started.",
                    interactive=False,
                    lines=3,
                )

        def on_settings_change(enabled_val, gpu_labels):
            global _enabled, _worker_gpu_ids
            _enabled = bool(enabled_val)
            _worker_gpu_ids = [int(g.split()[-1]) for g in (gpu_labels or [])]

        enabled.change(fn=on_settings_change, inputs=[enabled, selected_gpus], outputs=[])
        selected_gpus.change(fn=on_settings_change, inputs=[enabled, selected_gpus], outputs=[])

        def prewarm(enabled_val, selected):
            if not enabled_val:
                return "Multi-GPU is disabled."
            selected_ids = [int(g.split()[-1]) for g in (selected or [])]
            if not selected_ids:
                return "No worker GPUs selected."
            lines = []
            for dev_id in selected_ids:
                info(f"Pre-warming worker GPU {dev_id}...")
                w = get_or_start_worker(dev_id)
                status = "ready" if w["ready"] else "FAILED to start"
                lines.append(f"GPU {dev_id}: {status}")
            return "\n".join(lines)

        def kill_workers():
            info("Killing all worker processes...")
            shutdown_all()
            ok("All workers stopped.")
            return "All workers stopped."

        prewarm_btn.click(fn=prewarm, inputs=[enabled, selected_gpus], outputs=[status_box])
        kill_btn.click(fn=kill_workers, inputs=[], outputs=[status_box])

        return [enabled, selected_gpus]
