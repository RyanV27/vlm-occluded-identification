
## Description

This codebase is used to run the experiments for testing the capability of VLMs to identify occluded objects in synthetically-occluded COCO images. The largest object in a given COCO image is artificially occluded (via noise or texture patches) at varying severity levels (`no`, `low`, `medium`, `high`). The VLM is prompted once per image ("List everything that is visible in the image."), and evaluation checks whether the model's generation names the specific occluded object at each severity level.

## File structure

```
coco_occlusion/
├── models/                                  # definitions of functions for calling individual VLMs -> models/README.md
├── runs/                                    # output of the experiments gets stored here
├── evaluation_config.py                     # COCO category alias/synonym lists used when scoring model outputs
├── call_vlm.py                              # prompts a VLM on each occlusion level's images, saves generations
├── call_vlm_on_all_oc_types.sh              # orchestrates call_vlm.py -> calculate_alias_results.py per model
├── calculate_alias_results.py               # scores generations: did the model name the occluded object?
├── plot_comparative_results_individual.py   # plots results per experiment, pretrained vs finetuned, one graph per model
└── plot_comparative_results_together.py     # plots results per experiment, pretrained vs finetuned, all models on one graph
```

## Python Environment

This codebase reuses the same conda environments set up for `real_world_occlusion`. See the [real_world_occlusion README](../real_world_occlusion/README.md) for environment setup instructions, including notes for InternVL_3.5, Molmo2, and MiniCPM.

## Supported models (`--model_name`)

`Qwen2.5-VL`, `InternVL3.5`, `Molmo2`, `FT-Qwen2.5-VL`, `Gemma3`, `Llama-3.2`, `FT-Llama-3.2`, `FT-Gemma3`, `Qwen3-VL`, `FT-Qwen3-VL`

## Expected data directory structure

The directory passed via `--data_path` must be laid out with one subdirectory per occlusion level, each containing that level's images plus a `metadata.json`:

```
<data_path>/
├── no/
│   ├── metadata.json
│   ├── 1.jpg
│   └── ...
├── low/
│   ├── metadata.json
│   └── ...
├── medium/
│   ├── metadata.json
│   └── ...
└── high/
    ├── metadata.json
    └── ...
```

Each entry in a `metadata.json` corresponds to one scene and includes:

- `image_id` — unique scene identifier, used as the key in the generations output
- `file_path` — path to the scene's image; only the filename is used, resolved relative to that occlusion level's directory
- `occluded_category` — ground-truth label of the artificially-occluded object, used for scoring
- `occluder_category`, `categories_in_image` — additional ground-truth object labels, also used for scoring

## Commands to run

Prompt the VLM (writes `<out_path>/<model_name>/<model_name>_<occ_level>_generations.json`, resumable/checkpointed every 10 scenes):

```
python call_vlm.py --model_name "<model_name>" --occ_level "<occ_level>" --out_path "./runs/" --data_path "<path_to_coco_occlusion_data>"
```

e.g.

```
python call_vlm.py --model_name "Qwen2.5-VL" --occ_level "low" --out_path "./runs/"
```

Score the generations (writes `<out_path>/<model_name>/<model_name>_results.json`):

```
python calculate_alias_results.py --model_name "<model_name>" --occ_level "<occ_level>" [<occ_level> ...] --out_path "./runs/"
```

e.g.

```
python calculate_alias_results.py --model_name "Qwen2.5-VL" --occ_level "low" "medium" "high" --out_path "./runs/" > qwen_output.txt
```

Both scripts also accept `--model_cache_path` (default `/scratch/rsvargh2/huggingface_models/`) to control where Hugging Face weights are cached. Scoring uses the COCO category synonym lists in `evaluation_config.py` (`ALIASES`) to match model-generated object names against the ground-truth `occluded_category`.

## Automated pipeline

`call_vlm_on_all_oc_types.sh` runs both commands above back-to-back for a list of models across occlusion levels, activating the right conda env for each. Before running it, set these variables at the top of the script:

- `MODEL_NAMES` — array of model names to run; uncomment/edit the ones you want
- `OUT_PATH` — output directory for generations/results (e.g. `./runs/run2`)
- `OCC_LEVELS` — occlusion levels to prompt the VLM on (e.g. `("no" "low")`)
- `ALL_OCC_LEVELS` — occlusion levels to score once prompting is done (typically all four: `"no" "low" "medium" "high"`)

Conda env selection is automatic based on model name: `internvl_venv` for InternVL3.5, `molmo_venv` for Molmo2, `vqa_clutter_venv` otherwise — see the [real_world_occlusion README](../real_world_occlusion/README.md#notes-for-environment-setup-for-some-of-the-other-models-tested) for env setup notes.
