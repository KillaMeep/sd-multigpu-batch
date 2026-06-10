import copy
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests


def dispatch_all(p, work_splits: list, worker_ports: list, seeds: list, local_fn=None) -> list:
    """
    Dispatch sub-batches to GPUs in parallel.

    Args:
        p:            The original StableDiffusionProcessing object.
        work_splits:  List of (n_iter, batch_size) tuples, one per GPU slot.
                      Index 0 = local primary, 1+ = remote workers.
        worker_ports: Port per GPU slot (None = local, int = remote API port).
        seeds:        Starting seed per GPU slot.
        local_fn:     Callable for the local GPU slot. Pass the *original*
                      (unpatched) process_images to avoid re-entering any monkey-patch.

    Returns:
        List of results parallel to work_splits / worker_ports:
          - slot 0: Processed object (local GPU)
          - slot N: dict {"images": [...], "info": "..."} (remote worker)
          - None   if that slot was skipped (n_iter == 0) or failed.
    """
    from .payload_builder import build_payload
    from .log import err

    if local_fn is None:
        from modules.processing import process_images
        local_fn = process_images

    futures = {}
    results = [None] * len(work_splits)

    active = sum(1 for ni, _ in work_splits if ni > 0)
    with ThreadPoolExecutor(max_workers=active) as ex:
        for i, ((ni, bs), seed, port) in enumerate(zip(work_splits, seeds, worker_ports)):
            if ni == 0:
                continue

            if port is None:
                p2 = copy.copy(p)
                p2.n_iter = ni
                p2.batch_size = bs
                p2.seed = seed
                p2.extra_generation_params = copy.copy(p.extra_generation_params or {})
                p2.override_settings = copy.copy(p.override_settings or {})
                p2.override_settings["return_grid"] = False  # avoid partial-batch grid
                p2.comments = {}
                futures[ex.submit(local_fn, p2)] = i
            else:
                payload = build_payload(p, n_iter=ni, batch_size=bs, seed=seed)
                futures[ex.submit(_post, port, payload)] = i

        for future in as_completed(futures):
            idx = futures[future]
            try:
                results[idx] = future.result()
            except Exception as e:
                err(f"GPU slot {idx} failed: {e}")

    return results


def _post(port: int, payload: dict) -> dict:
    r = requests.post(
        f"http://127.0.0.1:{port}/sdapi/v1/txt2img",
        json=payload,
        timeout=600,
    )
    r.raise_for_status()
    return r.json()
