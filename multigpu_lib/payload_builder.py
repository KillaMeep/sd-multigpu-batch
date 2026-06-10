def build_payload(p, n_iter: int, seed: int, batch_size: int = None) -> dict:
    """
    Convert a StableDiffusionProcessing object into the JSON payload for
    POST /sdapi/v1/txt2img.  Only sends fields that are non-None.
    """
    # Merge any override_settings from p, then disable grid in the response.
    override = dict(p.override_settings or {})
    override["return_grid"] = False  # keep worker response clean (no grid image)

    payload = {
        "prompt": p.prompt or "",
        "negative_prompt": p.negative_prompt or "",
        "styles": list(p.styles or []),
        "seed": seed,
        "subseed": p.subseed,
        "subseed_strength": p.subseed_strength,
        "seed_resize_from_h": p.seed_resize_from_h,
        "seed_resize_from_w": p.seed_resize_from_w,
        "batch_size": batch_size if batch_size is not None else p.batch_size,
        "n_iter": n_iter,
        "steps": p.steps,
        "cfg_scale": p.cfg_scale,
        "width": p.width,
        "height": p.height,
        # bool fields: convert None → False
        "restore_faces": bool(p.restore_faces),
        "tiling": bool(p.tiling),
        # Workers should not save anything; the merged result handles presentation.
        "save_images": False,
        "send_images": True,
        "override_settings": override,
        "override_settings_restore_afterwards": True,
    }

    # Include optional sampler/scheduler fields only when set
    for key, attr in [
        ("sampler_name", "sampler_name"),
        ("scheduler", "scheduler"),
        ("eta", "eta"),
        ("s_churn", "s_churn"),
        ("s_tmax", "s_tmax"),
        ("s_tmin", "s_tmin"),
        ("s_noise", "s_noise"),
        ("s_min_uncond", "s_min_uncond"),
        ("refiner_checkpoint", "refiner_checkpoint"),
        ("refiner_switch_at", "refiner_switch_at"),
    ]:
        val = getattr(p, attr, None)
        if val is not None:
            payload[key] = val

    # High-res fix fields (txt2img only)
    if getattr(p, "enable_hr", False):
        payload["enable_hr"] = True
        payload["denoising_strength"] = getattr(p, "denoising_strength", 0.75) or 0.75
        payload["hr_scale"] = getattr(p, "hr_scale", 2.0)
        payload["hr_second_pass_steps"] = getattr(p, "hr_second_pass_steps", 0)
        payload["hr_resize_x"] = getattr(p, "hr_resize_x", 0)
        payload["hr_resize_y"] = getattr(p, "hr_resize_y", 0)
        for key, attr in [
            ("hr_upscaler", "hr_upscaler"),
            ("hr_checkpoint_name", "hr_checkpoint_name"),
            ("hr_sampler_name", "hr_sampler_name"),
            ("hr_scheduler", "hr_scheduler"),
            ("hr_prompt", "hr_prompt"),
            ("hr_negative_prompt", "hr_negative_prompt"),
        ]:
            val = getattr(p, attr, None)
            if val is not None:
                payload[key] = val

    return payload
