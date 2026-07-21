## Description

This pipeline runs the experiment where a cluttered tabletop scene contains a **single occluded object**. Each scene has four images, one per occlusion severity (`no`, `low`, `medium`, `high`), where a `front_obj` occludes a target `back_obj`. The VLM is prompted once per image ("List the objects on the table."), and evaluation checks whether the model's generation names the specific occluded object at each severity level.

## File structure

```
exp2/
├── call_vlm.py                        # prompts a VLM on each scene's 4 occlusion-level images, saves generations
├── evaluation_config.py               # object-name ALIASES dict + comparison_prompt template
├── exp2_calculate_alias_results.py    # scores generations: did the model name the occluded object?
└── exp2_run_eval_pipeline.sh          # orchestrates call_vlm.py -> exp2_calculate_alias_results.py per model
```

## Supported models (`--model_name`)

`Qwen2.5-VL`, `InternVL3.5`, `Molmo2`, `FT-Qwen2.5-VL`, `Gemma3`, `Llama-3.2`, `FT-Llama-3.2`, `FT-Gemma3`, `Qwen3-VL`, `FT-Qwen3-VL`, `COCO-FT-Llama-3.2`, `COCO-FT-Gemma3`, `COCO-FT-Qwen2.5-VL`, `COCO-FT-Qwen3-VL`

## Expected data directory structure

The directory passed via `--data_path` (and `DATA_PATH` in `exp2_run_eval_pipeline.sh`) must be laid out as:

```
<data_path>/
├── images/
│   ├── <occlusion_level>/
│   │   ├── 1.jpg
│   │   ├── 2.jpg
│   │   └── ...
│   ├── <occlusion_level>/
│   │   ├── 1.jpg
│   │   └── ...
│   └── ...
└── data.json
```

## Commands to run

Prompt the VLM (writes `<out_path>/<model_name>_generations.json`, resumable/checkpointed every 10 scenes):

```
python call_vlm.py --model_name "<model_name>" --out_path "./runs/exp2/" --data_path "<path_to_exp2_data>"
```

Score the generations (writes `<out_path>/identify_occluded_object/<model_name>_results.json`):

```
python exp2_calculate_alias_results.py --model_name "<model_name>" --out_path "./runs/exp2/"
```

Both scripts also accept `--model_cache_path` (default `/scratch/rsvargh2/huggingface_models/`) to control where Hugging Face weights are cached.

## Automated pipeline

`exp2_run_eval_pipeline.sh` runs both commands above back-to-back for a list of models, activating the right conda env for each. Before running it, set these variables at the top of the script:

- `OUT_PATH` — output directory for generations/results (e.g. `.../runs/exp2/run10`)
- `DATA_PATH` — path to the exp2 image/data directory (e.g. `/scratch/rsvargh2/occlusion_images/exp2`)
- `LOG_FILE` — path to the file where per-model command errors get logged (cleared at the start of each run)
- `MODEL_NAMES` — array of model names to run; uncomment/edit the ones you want (only `COCO-FT-Llama-3.2` and `COCO-FT-Gemma3` are uncommented by default)

Conda env selection is automatic based on model name: `internvl_venv` for InternVL3.5, `molmo_venv` for Molmo2, `vqa_clutter_venv` otherwise — see the [top-level README](../README.md#notes-for-environment-setup-for-some-of-the-other-models-tested) for env setup notes.

Results from repeated runs (`run1`-`run10`) are aggregated by `../runs/exp2/aggregate.py`.
