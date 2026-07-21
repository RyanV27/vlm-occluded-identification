## Description

Each file in this directory defines a class wrapping a single VLM (or, for the finetuned variants, a LoRA-finetuned checkpoint of one). Every class exposes:

- `__init__(cache_dir=...)` — loads the model/processor/tokenizer, defaulting the Hugging Face cache to `/scratch/rsvargh2/huggingface_models/`
- `run_inference(prompt, image_path)` — runs the model on one image and returns `(output_text, generated_ids, input_ids)`

For **base** model files, `cache_dir` can be set to a custom location to control where the (large) pretrained weights get downloaded/cached — otherwise it should be left unset/removed so the model falls back to the default Hugging Face cache folder. These models take up a lot of disk space, so it is advisable to explicitly set `cache_dir` to a location with enough room.

For **`finetuned_*.py`** and **`coco_ft_*.py`** files, `__init__()` instead takes a `model_save_dir`/`model_save_path` argument (naming varies by file), which must be set to the location of that model's finetuned checkpoint. Each file defaults this to the checkpoint path used for this project's own runs, so it should be updated to point at wherever your checkpoint actually lives.

## Naming convention

There are three tiers per model architecture:

- **Base** (e.g. `qwen2_5VL.py`, `gemma_3.py`, `llama3_2.py`) — loads the vanilla pretrained checkpoint straight from the Hugging Face Hub.
- **`finetuned_*.py`** — loads a LoRA checkpoint (via Unsloth) finetuned on this project's own real-world occlusion tabletop data.
- **`coco_ft_*.py`** — loads a LoRA checkpoint (via Unsloth) finetuned on COCO-based synthetic occlusion data.

## File structure

| File | Model |
|---|---|
| `qwen2_5VL.py` | Qwen2.5-VL (base) |
| `finetuned_qwen2_5VL.py` | Qwen2.5-VL, finetuned on real-world occlusion data |
| `coco_ft_qwen2_5VL.py` | Qwen2.5-VL, finetuned on COCO synthetic occlusion data |
| `qwen3VL.py` | Qwen3-VL (base) |
| `finetuned_qwen3VL.py` | Qwen3-VL, finetuned on real-world occlusion data |
| `coco_ft_qwen3VL.py` | Qwen3-VL, finetuned on COCO synthetic occlusion data |
| `gemma_3.py` | Gemma3 (base) |
| `finetuned_gemma_3.py` | Gemma3, finetuned on real-world occlusion data |
| `coco_ft_gemma_3.py` | Gemma3, finetuned on COCO synthetic occlusion data |
| `gemma_3n.py` | Gemma3n (base) |
| `finetuned_gemma_3n.py` | Gemma3n, finetuned on real-world occlusion data |
| `llama3_2.py` | Llama-3.2 (base) |
| `finetuned_llama3_2.py` | Llama-3.2, finetuned on real-world occlusion data |
| `coco_ft_llama3_2.py` | Llama-3.2, finetuned on COCO synthetic occlusion data |
| `internVL3_5.py` | InternVL3.5 (base) |
| `molmo.py` | Molmo (base) |
| `molmo2.py` | Molmo2 (base) |
| `kimi_VL_A3B.py` | Kimi-VL-A3B (base) |
| `minicpm_o_2_6.py` | MiniCPM-o-2.6 (base) |
| `ministral_3.py` | Ministral-3 (base) |

## `__init__.py`

Model classes are imported into `__init__.py` individually. Only uncomment the import for the model(s) you actually need for a given run — different models (e.g. InternVL3.5, Molmo2) require different, sometimes conflicting, package versions, so uncommenting everything at once can break the active virtual environment. See the [top-level README](../README.md#notes-for-environment-setup-for-some-of-the-other-models-tested) for per-model environment setup notes.
