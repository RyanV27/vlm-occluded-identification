
## Description

This codebase is used to run the experiments for testing the capability of VLMs to identify occluded objects on a cluttered tabletop. 

## File structure

```
real_world_occlusion/
├── exp2/                                    # pipeline for cluttered tabletop w/ a single occluded object -> exp2/README.md
├── exp3/                                    # pipeline for cluttered tabletop w/ multiple occluded objects -> exp3/README.md
├── models/                                  # definitions of functions for calling individual VLMs -> models/README.md
├── runs/                                    # output of the experiments gets stored here
├── plot_comparative_results_individual.py   # plots results per experiment, pretrained vs finetuned, one graph per model
├── plot_comparative_results_together.py     # plots results per experiment, pretrained vs finetuned, all models on one graph
└── requirements.txt                         # base Python package requirements
```

## Python Environment

I used conda to create python environments to run experiments for each model. Some models needed their own custom environments, especially the Hugging Face models, as they needed specific Python packages and versions.

After creating a Python virtual environment, the below command can be used to install the packages required to run unsloth and Qwen Hugging Face models.

```
pip install -r requirements.txt
```

### Notes for environment setup for some of the other models tested:

**InternVL_3.5** — requires `transformers<5`, `decord`, `einops`, `timm`, `qwen_vl_utils`.

**Molmo2** — conda environment should be set up as:

```
conda create --name molmo_venv python=3.11
conda activate molmo_venv
pip install transformers==4.57.1
pip install torch pillow einops torchvision accelerate decord2 molmo_utils
```

**MiniCPM** — requires the following additional packages: `soundfile`, `librosa`, `torchaudio`.

```
pip install \
  "torch==2.3.1" "torchvision==0.18.1" "torchaudio==2.3.1" \
  "transformers==4.44.2" "sentencepiece==0.2.0" "accelerate==1.2.1"
```