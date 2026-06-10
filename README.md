# Multi-GPU Batch — AUTOMATIC1111 Extension

Transparently splits image generation across multiple GPUs. You submit a normal txt2img or img2img request; the extension divides the work across all selected GPUs, runs them in parallel, and returns a single merged result — no change to your normal workflow.

## How it works

A1111 is single-GPU per process by design. This extension works around that by spawning headless worker A1111 processes (one per additional GPU) via `--nowebui --api --device-id N`, dispatching sub-batches to each via the REST API, and merging the results back into a single `Processed` response.

```
Primary process (GPU 0)         Worker process (GPU 1)
  process_images(p, n=2)          POST /sdapi/v1/txt2img  (n=2)
        │                                   │
        └──────────── parallel ─────────────┘
                          │
                    merge_results()
                          │
                   gallery (4 images)
```

Work is split at the image level: `batch_count × batch_size` images are distributed as evenly as possible, so both `batch_count=4 / batch_size=1` and `batch_count=1 / batch_size=4` are handled correctly.

Seeds are assigned to preserve the same sequence you would have gotten from a single-GPU run, so results are reproducible.

## Requirements

- AUTOMATIC1111 WebUI (tested on v1.10.x)
- 2+ CUDA GPUs visible to the same machine
- No additional Python packages (uses only what A1111 already ships)

## Installation

### Via the Extensions tab (recommended)

1. Open A1111 → **Extensions** → **Install from URL**
2. Paste the repo URL and click **Install**
3. Restart A1111

### Manual

```bash
cd <a1111_root>/extensions
git clone https://github.com/KillaMeep/sd-multigpu-batch
```

Restart A1111. No additional `pip install` is needed.

## Usage

1. Open the **Multi-GPU Batch** accordion (always visible on txt2img and img2img tabs)
2. Check **Enable Multi-GPU splitting**
3. Select the worker GPUs you want to use (GPU 0 always runs locally)
4. *(Optional)* Click **Pre-warm Workers** to start the worker process now — otherwise it starts on the first generation and adds a one-time 30–90 s startup delay
5. Generate as normal

![UI screenshot placeholder](docs/screenshot.png)

### Worker status

The status box in the accordion shows whether each worker is ready. Workers stay alive across generations and are shut down automatically when A1111 exits.

Click **Kill Workers** to terminate all worker processes immediately (e.g. to free VRAM).

## Console output

The extension logs all activity to the server console, colour-coded:

| Colour | Meaning |
|--------|---------|
| Cyan tag `[MULTIGPU]` | All messages |
| Green text | Success / ready |
| Yellow text | Warnings |
| Red text | Errors / failures |

Example:

```
[MULTIGPU] Extension loaded — 2 CUDA device(s) detected. process_images patched.
[MULTIGPU] Distributing 4 image(s) across 2 GPU(s):
[MULTIGPU]   GPU 0 (local): 2 iter(s) × batch 1  seed=1234567890
[MULTIGPU]   GPU 1 (port 34521): 2 iter(s) × batch 1  seed=1234567892
[MULTIGPU] Dispatching — all GPUs running in parallel...
[MULTIGPU] Done — merged 4 image(s) from 2 GPU(s).
```

## Configuration

No config file is needed. Worker ports are chosen randomly from the ephemeral range (20000–55000) with a real `bind()` check, so they never conflict with the primary or each other.

The model and key options (`sd_model_checkpoint`, `sd_vae`, `CLIP_stop_at_last_layers`, `eta_noise_seed_delta`) are synced from the primary to each worker before every generation, so model changes in the UI are picked up automatically.

## Known limitations

- **Grid images are suppressed.** Each GPU generates its sub-batch without a grid; no merged grid is regenerated. You get individual images only.
- **Extensions on workers.** Workers are headless API processes — no other extensions run on them. If another extension modifies generation (e.g. a custom sampler), workers won't have it.
- **img2img.** The dispatcher payload builder does not yet include `init_images`, so img2img will fall back to single-GPU.
- **First-run latency.** The first generation after a cold start waits ~30–90 s for the worker to load the model. Pre-warm to avoid this.
- **Single machine only.** Workers are spawned as local subprocesses; distributed/networked setups are not supported.

## File structure

```
extensions/multigpu-batch/
  scripts/
    multigpu_batch.py       # A1111 Script class — UI + process_images patch
  multigpu_lib/
    worker_manager.py       # Subprocess lifecycle, health check, model sync
    dispatcher.py           # Parallel dispatch via ThreadPoolExecutor
    payload_builder.py      # StableDiffusionProcessing → REST API payload
    result_merger.py        # Merge Processed + remote worker dicts
    seed_utils.py           # Image-level work splitting + seed calculation
    log.py                  # Coloured console logger
  install.py                # No extra dependencies
  README.md
```
