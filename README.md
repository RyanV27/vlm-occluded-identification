# Improving Occluded Object Identification Capability of VLMs by Finetuning

This codebase evaluates how well vision-language models (VLMs) identify occluded objects in cluttered scenes, in two settings: real-world tabletop photos and synthetically-occluded COCO images. It also explores whether LoRA finetuning on synthetically-occluded images improves a model's ability to name occluded objects.

## File structure

```
codebase/
├── real_world_occlusion/   # experiments on real-world cluttered tabletop photos -> real_world_occlusion/README.md
├── coco_occlusion/         # experiments on synthetically-occluded COCO images   -> coco_occlusion/README.md
└── fine_tuning/            # LoRA finetuning of VLMs + generating COCO occlusion data for finetuning -> fine_tuning/README.md
```

## Subprojects

### `real_world_occlusion/`

Tests VLMs on real-world photos of a cluttered tabletop. Two sub-experiments:

- **exp2** — a single occluded object per scene, imaged at four occlusion severities (`no`, `low`, `medium`, `high`).
- **exp3** — multiple occluded objects in a single scene image, scored per severity bucket.

In both, the VLM is prompted to list everything visible on the table, and evaluation checks whether the occluded object(s) are named. See [real_world_occlusion/README.md](real_world_occlusion/README.md), [exp2/README.md](real_world_occlusion/exp2/README.md), and [exp3/README.md](real_world_occlusion/exp3/README.md).

### `coco_occlusion/`

Tests VLMs on COCO images where the largest object in each image is artificially occluded (via noise or texture patches) at varying severities. The VLM is prompted to list everything visible, and evaluation checks whether the occluded object is named. See [coco_occlusion/README.md](coco_occlusion/README.md).

### `fine_tuning/`

Notebooks/scripts for LoRA finetuning (via [Unsloth](https://unsloth.ai/docs/basics/vision-fine-tuning)) of the VLMs used above, on either COCO or real-world tabletop data, plus the script that generates the synthetically-occluded COCO training images. See [fine_tuning/README.md](fine_tuning/README.md).

## Models tested

Qwen2.5-VL, Qwen3-VL, Gemma3 (and Gemma3n), Llama-3.2, InternVL3.5, Molmo / Molmo2, MiniCPM-o-2.6, Kimi-VL-A3B, and finetuned (LoRA) variants of the first four on both real-world and COCO occlusion data. Each `models/` subdirectory (in `real_world_occlusion/` and `coco_occlusion/`) wraps these in a common `run_inference(prompt, image_path)` interface — see their respective `models/README.md` for the full list and setup notes.

## Getting started

1. Set up a Python/conda environment — see [real_world_occlusion/README.md](real_world_occlusion/README.md#python-environment) for base requirements and per-model notes (InternVL3.5, Molmo2, MiniCPM need extra/conflicting packages, so separate environments per model are recommended). `coco_occlusion/` and `fine_tuning/` reuse these same environments.
2. Pick a subproject (`real_world_occlusion/exp2`, `real_world_occlusion/exp3`, or `coco_occlusion/`) and follow its README for expected data layout and commands to prompt/score a model.
3. To finetune a model on your own data first, see `fine_tuning/README.md`.

Across all subprojects, results are written under each project's `runs/` directory and aggregated/plotted with the `plot_comparative_results_*.py` scripts (pretrained vs. finetuned, per model or all models together).
