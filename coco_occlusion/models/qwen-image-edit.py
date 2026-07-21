import os
from PIL import Image
import torch

from diffusers import QwenImageEditPipeline

pipeline = QwenImageEditPipeline.from_pretrained("Qwen/Qwen-Image-Edit", cache_dir="/scratch/rsvargh2/huggingface_models/")
print("pipeline loaded")
pipeline.to(torch.bfloat16)
pipeline.to("cuda")
pipeline.set_progress_bar_config(disable=None)

image = Image.open("/scratch/rsvargh2/occlusion_images/images/high/25.jpg").convert("RGB")
prompt = "Remove the object in the front."

inputs = {
    "image": image,
    "prompt": prompt,
    "generator": torch.manual_seed(0),
    "true_cfg_scale": 4.0,
    "negative_prompt": " ",
    "num_inference_steps": 50,
}

with torch.inference_mode():
    output = pipeline(**inputs)
    output_image = output.images[0]
    output_image.save(os.path.abspath("runs/output_image_edit.jpg"))
    print("image saved at", os.path.abspath("runs/output_image_edit.jpg"))