"""
generate_occluded_scenes.py

Loads random images from the COCO val2017 dataset, artificially occludes the
largest object in each image, and saves the results with metadata.

Usage:
    python generate_occluded_scenes_noise.py --level low --mode noise --count 5000 --action write
    python generate_occluded_scenes_noise.py --level high --mode texture --texture path/to/texture.jpg --count 3 --action append

Required folder structure:
    augmenting_coco_images/
        val2017/                          <- images (already downloaded)
        annotations/
            instances_val2017.json        <- instance segmentation annotations
            captions_val2017.json         <- image captions
"""

import os
import json
import math
import random
import argparse
import numpy as np
from pycocotools.coco import COCO
from PIL import Image

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
# BASE_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = "/scratch/rsvargh2/coco/"
IMG_DIR = os.path.join(BASE_DIR, "train2017")
ANN_INSTANCES = os.path.join(BASE_DIR, "annotations", "instances_train2017.json")
ANN_CAPTIONS = os.path.join(BASE_DIR, "annotations", "captions_train2017.json")

LEVEL_RANGES = {
    "low":    (15, 25),
    "medium": (40, 50),
    "high":   (65, 75),
}

# Maximum number of images to try before giving up on a single output image.
MAX_IMAGE_RETRIES = 20

# The target (occluded) annotation must cover at least this fraction of the image area.
MIN_TARGET_IMAGE_FRACTION = 0.05   # 5 %


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_cocos():
    """Load COCO instances and captions APIs."""
    for path, label in [(ANN_INSTANCES, "Instances"), (ANN_CAPTIONS, "Captions")]:
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"{label} annotation file not found:\n  {path}\n"
                "Download and unzip:\n"
                "  http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
            )
    print("Loading COCO instances annotations...")
    print(f"ann instances: {ANN_INSTANCES}")
    coco_inst = COCO(ANN_INSTANCES)
    print("Loading COCO captions annotations...")
    coco_caps = COCO(ANN_CAPTIONS)
    return coco_inst, coco_caps


def pick_image_with_single_instance_category(coco_inst, used_ids):
    """
    Return a random image where at least one category has exactly one annotation
    (other categories may have multiple instances).  The returned target_ann is
    chosen from those single-instance categories and will be the object occluded.
    The target annotation must cover at least MIN_TARGET_IMAGE_FRACTION of the image.

    Returns (img_id, target_ann, all_valid_anns).
    """
    from collections import Counter

    all_img_ids = coco_inst.getImgIds()
    random.shuffle(all_img_ids)

    for img_id in all_img_ids:
        if img_id in used_ids:
            continue

        ann_ids = coco_inst.getAnnIds(imgIds=img_id, iscrowd=False)
        anns = coco_inst.loadAnns(ann_ids)
        valid_anns = [a for a in anns if a.get("segmentation") and len(a["segmentation"]) > 0]

        if not valid_anns:
            continue

        img_info = coco_inst.loadImgs(img_id)[0]
        img_area = img_info["width"] * img_info["height"]

        cat_counts = Counter(a["category_id"] for a in valid_anns)
        single_instance_anns = [
            a for a in valid_anns
            if cat_counts[a["category_id"]] == 1
            and a["area"] / img_area >= MIN_TARGET_IMAGE_FRACTION
        ]

        if not single_instance_anns:
            continue

        target_ann = random.choice(single_instance_anns)
        return img_id, target_ann, valid_anns

    raise RuntimeError("Could not find a suitable image with a single-instance category.")


# ---------------------------------------------------------------------------
# Occlusion helpers
# ---------------------------------------------------------------------------
def build_occluder(bbox_crop, mode, texture_img=None):
    """Return an RGB uint8 numpy array matching bbox_crop's shape for the given mode."""
    if mode == "black":
        return np.zeros_like(bbox_crop)
    elif mode == "white":
        return np.full_like(bbox_crop, 255)
    elif mode == "noise":
        return np.random.randint(0, 256, bbox_crop.shape, dtype=np.uint8)
    elif mode == "texture":
        h, w = bbox_crop.shape[:2]
        tex_pil = Image.fromarray(texture_img).resize((w, h), Image.LANCZOS)
        return np.array(tex_pil)


def random_occluding_box(mask, pct_low, pct_high, other_masks=(), max_attempts=3000):
    """
    Find a random rectangle in full-image coordinates whose overlap with `mask`
    is between pct_low and pct_high percent of the total mask pixels, and that
    does not fully cover any mask in `other_masks`.

    Returns (x1, y1, x2, y2) or raises RuntimeError if not found.
    """
    h, w = mask.shape
    object_pixels = int(mask.sum())
    if object_pixels == 0:
        raise RuntimeError("Object mask is empty.")

    # Tight bounding box of the object mask
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    bbox_h = max(rmax - rmin + 1, 1)
    bbox_w = max(cmax - cmin + 1, 1)
    bbox_area = bbox_h * bbox_w

    # Estimate box size targeting the midpoint occlusion %
    target_mid = (pct_low + pct_high) / 2 / 100
    density = object_pixels / bbox_area          # fraction of bbox that is object
    # area of box needed so that (box_area * density) ≈ target_mid * object_pixels
    estimated_area = target_mid * object_pixels / max(density, 1e-6)
    base_side = math.sqrt(estimated_area)

    for _ in range(max_attempts):
        # Jitter the box size ±30%
        jitter = random.uniform(0.7, 1.3)
        bh = min(int(base_side * jitter), h)
        bw = min(int(base_side * jitter), w)
        bh = max(bh, 1)
        bw = max(bw, 1)

        # Randomly position the box anchored somewhere inside/around the object bbox
        y_lo = max(0, rmin - bh // 2)
        y_hi = max(y_lo, min(h - bh, rmax))
        y1 = random.randint(y_lo, y_hi)

        x_lo = max(0, cmin - bw // 2)
        x_hi = max(x_lo, min(w - bw, cmax))
        x1 = random.randint(x_lo, x_hi)
        y2 = min(y1 + bh, h)
        x2 = min(x1 + bw, w)

        overlap = int(mask[y1:y2, x1:x2].sum())
        pct = overlap / object_pixels * 100

        if not (pct_low <= pct <= pct_high):
            continue

        # Reject if the box completely covers any other annotation
        fully_covers = any(
            int(om.sum()) > 0 and int(om[y1:y2, x1:x2].sum()) == int(om.sum())
            for om in other_masks
        )
        if fully_covers:
            continue

        return x1, y1, x2, y2

    raise RuntimeError(
        f"Could not find a valid box achieving {pct_low}–{pct_high}% occlusion "
        f"without fully covering another annotation after {max_attempts} attempts."
    )


def apply_occlusion(img_array, mask, box_coords, mode, texture_img=None):
    """
    Apply the occluder pattern to the entire box_coords region (not just the mask).
    The occlusion percentage is still measured as the fraction of mask pixels
    covered by the box.

    Returns (occluded_img_array, pct_occ).
    """
    x1, y1, x2, y2 = box_coords
    bbox_crop = img_array[y1:y2, x1:x2].copy()
    occluder = build_occluder(bbox_crop, mode, texture_img)

    result = img_array.copy()
    result[y1:y2, x1:x2] = occluder

    overlap_mask = mask[y1:y2, x1:x2].astype(bool)
    pct_occ = overlap_mask.sum() / mask.sum() * 100
    return result, float(pct_occ)


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------
def save_metadata(entry, mode, level):
    """Append a metadata entry to occluded_images/{mode}/{level}/metadata.json."""
    meta_path = os.path.join(BASE_DIR, "occluded_images", mode, level, "metadata.json")
    if os.path.exists(meta_path):
        with open(meta_path, "r") as f:
            data = json.load(f)
    else:
        data = []
    data.append(entry)
    with open(meta_path, "w") as f:
        json.dump(data, f, indent=2)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Generate occluded COCO images.")
    parser.add_argument("--level", required=True, choices=["low", "medium", "high"],
                        help="Occlusion level: low (15-25%%), medium (40-50%%), high (65-75%%)")
    parser.add_argument("--mode", required=True, choices=["black", "white", "noise", "texture"],
                        help="Occluder pattern")
    parser.add_argument("--count", required=True, type=int,
                        help="Number of images to generate")
    parser.add_argument("--texture", default=None,
                        help="Path to texture image (required when --mode texture)")
    parser.add_argument("--action", required=True, choices=["write", "append"],
                        help="'write' clears existing metadata and overwrites images; 'append' adds to existing")
    args = parser.parse_args()

    if args.mode == "texture":
        if args.texture is None:
            parser.error("--texture is required when --mode is texture")
        if not os.path.exists(args.texture):
            parser.error(f"Texture file not found: {args.texture}")
        texture_img = np.array(Image.open(args.texture).convert("RGB"))
    else:
        texture_img = None

    pct_low, pct_high = LEVEL_RANGES[args.level]

    # Create output directory
    out_dir = os.path.join(BASE_DIR, "occluded_images", args.mode, args.level)
    os.makedirs(out_dir, exist_ok=True)

    # Write mode: clear existing metadata so this run starts fresh
    meta_path = os.path.join(out_dir, "metadata.json")
    if args.action == "write" and os.path.exists(meta_path):
        os.remove(meta_path)

    coco_inst, coco_caps = load_cocos()
    print("here")

    used_ids = set()

    for i in range(args.count):
        print(f"\n[{i+1}/{args.count}] Picking image...")

        # Try up to MAX_IMAGE_RETRIES images until a valid box placement is found.
        for image_attempt in range(1, MAX_IMAGE_RETRIES + 1):
            img_id, target_ann, anns = pick_image_with_single_instance_category(coco_inst, used_ids)
            used_ids.add(img_id)

            img_info = coco_inst.loadImgs(img_id)[0]
            img_path = os.path.join(IMG_DIR, img_info["file_name"])
            if not os.path.exists(img_path):
                print(f"  Image file missing ({img_info['file_name']}), trying next.")
                continue

            image = Image.open(img_path).convert("RGB")
            img_array = np.array(image)

            target_category = coco_inst.loadCats(target_ann["category_id"])[0]["name"]
            mask = coco_inst.annToMask(target_ann)

            # Build masks for all other annotations to guard against full coverage
            other_anns = [a for a in anns if a["id"] != target_ann["id"]]
            other_masks = [coco_inst.annToMask(a) for a in other_anns]

            print(f"  Image : {img_info['file_name']} (id={img_id})")
            print(f"  Target: {target_category} (area={target_ann['area']:.0f} px)")
            print(f"  Level : {args.level} ({pct_low}–{pct_high}%)")
            print(f"  Mode  : {args.mode}")

            try:
                box_coords = random_occluding_box(mask, pct_low, pct_high,
                                                  other_masks=other_masks)
            except RuntimeError as e:
                print(f"  {e}")
                if image_attempt < MAX_IMAGE_RETRIES:
                    print(f"  Discarding image, selecting a new one "
                          f"(attempt {image_attempt}/{MAX_IMAGE_RETRIES})...")
                continue  # pick a new image

            occluded_array, pct_occ = apply_occlusion(img_array, mask, box_coords,
                                                      args.mode, texture_img)
            break  # valid box found — proceed with this image
        else:
            raise RuntimeError(
                f"Could not find a valid box after trying {MAX_IMAGE_RETRIES} images."
            )

        # Captions
        cap_ann_ids = coco_caps.getAnnIds(imgIds=img_id)
        captions = [c["caption"] for c in coco_caps.loadAnns(cap_ann_ids)]

        all_categories = [
            coco_inst.loadCats(a["category_id"])[0]["name"] for a in anns
        ]

        print(f"  Occlusion applied: {pct_occ:.1f}%")

        # Save image
        out_filename = img_info["file_name"]
        out_path = os.path.join(out_dir, out_filename)
        Image.fromarray(occluded_array).save(out_path)
        print(f"  Saved : {out_path}")

        # Save metadata
        rel_out = os.path.join("occluded_images", args.mode, args.level, out_filename).replace("\\", "/")
        rel_orig = os.path.join("val2017", img_info["file_name"]).replace("\\", "/")
        entry = {
            "file_path": rel_out,
            "original_file": rel_orig,
            "image_id": img_id,
            "captions": captions,
            "categories_in_image": all_categories,
            "occluded_category": target_category,
            "occlusion_level": args.level,
            "occlusion_mode": args.mode,
            "occlusion_pct": round(pct_occ, 2),
        }
        save_metadata(entry, args.mode, args.level)

        # Display
        # Image.fromarray(occluded_array).show(title=f"Occluded — {out_filename}")

        print("\nCaptions:")
        for j, cap in enumerate(captions, 1):
            print(f"  {j}. {cap}")

    print(f"\nDone. {args.count} image(s) saved to occluded_images/{args.mode}/{args.level}/")


if __name__ == "__main__":
    main()
