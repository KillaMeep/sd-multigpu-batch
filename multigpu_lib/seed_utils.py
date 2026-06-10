import random


def resolve_seed(seed: int) -> int:
    """Replace -1 with a random seed, leave other values unchanged."""
    return random.randint(0, 2**32 - 1) if seed == -1 else seed


def split_work(n_iter: int, batch_size: int, n_gpus: int) -> list:
    """
    Distribute total images (n_iter * batch_size) across n_gpus.
    Returns list of (n_iter, batch_size) tuples, one per GPU slot.

    When n_iter >= n_gpus: splits n_iter, each GPU keeps original batch_size.
    When n_iter < n_gpus:  splits at the image level so all GPUs get work;
                           each active GPU gets 1 iteration with a reduced batch_size.
    Slots that receive 0 images get (0, 0).
    """
    if n_iter >= n_gpus:
        base, rem = divmod(n_iter, n_gpus)
        return [
            (base + (1 if i < rem else 0), batch_size)
            for i in range(n_gpus)
        ]

    # Fewer iterations than GPUs — split at individual-image granularity.
    total = n_iter * batch_size
    base_imgs, rem = divmod(total, n_gpus)
    result = []
    for i in range(n_gpus):
        img_count = base_imgs + (1 if i < rem else 0)
        if img_count == 0:
            result.append((0, 0))
        else:
            # 1 iteration with img_count images; never exceeds original batch_size.
            result.append((1, img_count))
    return result


def split_seeds_from_work(base_seed: int, work_splits: list) -> list:
    """
    Return starting seed for each GPU given a list of (n_iter, batch_size) tuples.
    Preserves the contiguous seed sequence across all GPUs.
    """
    starts = []
    cursor = base_seed
    for n_iter, batch_size in work_splits:
        starts.append(cursor)
        cursor += n_iter * batch_size
    return starts


# ── Legacy helpers (kept for existing callers / tests) ───────────────────────

def split_n_iter(total: int, n_gpus: int) -> list:
    base, rem = divmod(total, n_gpus)
    return [base + (1 if i < rem else 0) for i in range(n_gpus)]


def split_seeds(base_seed: int, batch_size: int, n_iter_splits: list) -> list:
    starts = []
    cursor = base_seed
    for n in n_iter_splits:
        starts.append(cursor)
        cursor += n * batch_size
    return starts
