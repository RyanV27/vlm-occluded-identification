# VLLM code

import torch
from vllm import LLM
from PIL import Image

import torch
torch.backends.cuda.enable_flash_sdp(False)
torch.backends.cuda.enable_mem_efficient_sdp(False)
torch.backends.cuda.enable_math_sdp(True)

# Load the model with vLLM
llm = LLM(
    model="moonshotai/Kimi-VL-A3B-Instruct",
    enforce_eager=True,
    max_model_len=2048,
    trust_remote_code=True,  # required for models with custom code
    limit_mm_per_prompt={"image": 1},  # allow one image per prompt
    gpu_memory_utilization=0.3,
    dtype=torch.bfloat16
)

# Load an image from disk
image = Image.open("/scratch/rsvargh2/occlusion_images/images/no/1.jpg")

# Prepare your multimodal prompt
prompt = "Describe this image."

# Call generate() with multimodal data
outputs = llm.generate(
    [
        {
            "prompt": prompt,                        # text prompt
            "multi_modal_data": {"image": image},    # image input
        }
    ]
)

# Print the generated text
for out in outputs:
    print(out.outputs[0].text)