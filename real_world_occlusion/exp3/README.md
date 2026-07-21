## Description

This pipeline runs the experiment where a cluttered tabletop scene contains **multiple occluded objects**. Unlike exp2, each scene has a single image (no separate image per occlusion level). The VLM is prompted once per scene ("List the objects on the table."), and evaluation checks, for each occlusion-level bucket (`no`, `low`, `medium`, `high`), what percentage of that bucket's ground-truth objects were correctly named in the single generation.

## File structure

```
exp3/
├── call_vlm.py                        # prompts a VLM with each scene's single image, saves generations
├── evaluation_config.py               # object-name ALIASES dict + comparison_prompt template
├── exp3_calculate_alias_results.py    # scores generations against per-occlusion-level object lists
└── exp3_run_eval_pipeline.sh          # orchestrates call_vlm.py -> exp3_calculate_alias_results.py per model
```

## Supported models (`--model_name`)

`Qwen2.5-VL`, `InternVL3.5`, `Molmo2`, `Molmo`, `FT-Qwen2.5-VL`, `Gemma3n`, `Gemma3`, `Llama-3.2`, `FT-Llama-3.2`, `FT-Gemma3`, `Qwen3-VL`, `FT-Qwen3-VL`, `COCO-FT-Llama-3.2`, `COCO-FT-Gemma3`, `COCO-FT-Qwen2.5-VL`, `COCO-FT-Qwen3-VL`

## Expected data directory structure

The directory passed via `--data_path` (and `DATA_PATH` in `exp3_run_eval_pipeline.sh`) must be laid out as:

```
<data_path>/
├── images/
│   ├── 1.jpg
│   ├── 2.jpg
│   └── ...
└── data.json
```

Each entry in `data.json` is keyed by scene `id` and includes:

- `image_path` — path to the scene's image, relative to `--data_path`
- `no_occlusion_objects`, `low_occlusion_objects`, `medium_occlusion_objects`, `high_occlusion_objects` — ground-truth object lists for each occlusion-level bucket, used for scoring

## Commands to run

Prompt the VLM (writes `<out_path>/<model_name>_generations.json`, resumable/checkpointed every 10 scenes):

```
python call_vlm.py --model_name "<model_name>" --out_path "./runs/exp3/" --data_path "<path_to_exp3_data>"
```

Score the generations (writes `<out_path>/<model_name>_results.json`):

```
python exp3_calculate_alias_results.py --model_name "<model_name>" --out_path "./runs/exp3/"
```

Both scripts also accept `--model_cache_path` (default `/scratch/rsvargh2/huggingface_models/`) to control where Hugging Face weights are cached.

## Automated pipeline

`exp3_run_eval_pipeline.sh` runs both commands above back-to-back for a list of models, activating the right conda env for each. Before running it, set these variables at the top of the script:

- `OUT_PATH` — output directory for generations/results (e.g. `.../runs/exp3/run10`)
- `DATA_PATH` — path to the exp3 image/data directory (e.g. `/scratch/rsvargh2/occlusion_images/exp3`)
- `LOG_FILE` — path to the file where per-model command errors get logged (cleared at the start of each run)
- `MODEL_NAMES` — array of model names to run; uncomment/edit the ones you want (all four `COCO-FT-*` models are uncommented by default)

Conda env selection is automatic based on model name: `internvl_venv` for InternVL3.5, `molmo_venv` for Molmo2, `vqa_clutter_venv` otherwise — see the [top-level README](../README.md#notes-for-environment-setup-for-some-of-the-other-models-tested) for env setup notes.

Results from repeated runs (`run1`-`run10`) are aggregated by `../runs/exp3/aggregate.py`.
