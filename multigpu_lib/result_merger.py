import base64
import io
import json

from PIL import Image

from modules.processing import Processed

from .log import err


def merge_results(results: list) -> Processed:
    """
    Merge all generation results into a single Processed object.

    results[0] must be a Processed from the local (primary) GPU.
    results[1:] are dicts from remote worker API responses:
        {"images": [base64_str, ...], "info": "...json..."}

    Worker responses contain no grid image when save_images=False (the default),
    so every entry in images[] is a real output image.
    """
    primary: Processed = results[0]

    # Strip any grid image(s) prepended by process_images (index_of_first_image marks
    # where real images start; grids live before that index).
    first = getattr(primary, "index_of_first_image", 0)
    if first > 0:
        primary.images = primary.images[first:]
        primary.index_of_first_image = 0

    for result in results[1:]:
        if result is None:
            continue

        images_b64 = result.get("images", [])
        for img_b64 in images_b64:
            try:
                raw = base64.b64decode(img_b64)
                img = Image.open(io.BytesIO(raw)).convert("RGB")
                primary.images.append(img)
            except Exception as e:
                print(f"[MultiGPU] Failed to decode worker image: {e}")

        try:
            info = json.loads(result.get("info", "{}"))
            primary.infotexts.extend(info.get("infotexts", []))
            primary.all_prompts.extend(info.get("all_prompts", []))
            primary.all_seeds.extend(info.get("all_seeds", []))
            primary.all_subseeds.extend(info.get("all_subseeds", []))
            primary.all_negative_prompts.extend(info.get("all_negative_prompts", []))
        except Exception as e:
            print(f"[MultiGPU] Failed to merge worker infotexts: {e}")

    return primary
