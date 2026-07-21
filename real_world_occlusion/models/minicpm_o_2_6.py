# This code didn't work out as it requires installing flash_attn.
# I am restricted from installing flash attention.

import torch
from PIL import Image
from transformers import AutoModel, AutoTokenizer


class MiniCPMO26:
    def __init__(self, model_id="openbmb/MiniCPM-o-2_6", cache_dir="/scratch/rsvargh2/huggingface_models/"):
        self.model = AutoModel.from_pretrained(
            model_id,
            trust_remote_code=True,
            attn_implementation="sdpa",
            torch_dtype=torch.bfloat16,
            init_vision=True,
            init_audio=False,
            init_tts=False,
            cache_dir=cache_dir
        ).eval().cuda()

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_id, 
            trust_remote_code=True, 
            cache_dir=cache_dir
        )

    def run_inference(self, prompt, image_path):
        """
        Run inference on the model.

        Args:
            prompt     (str): Text prompt/question about the image.
            image_path (str): Path to the image file.

        Returns:
            str: The generated text response.
        """
        image = Image.open(image_path).convert('RGB')
        msgs = [{'role': 'user', 'content': [image, prompt]}]

        response = self.model.chat(
            image=None,
            msgs=msgs,
            tokenizer=self.tokenizer
        )
        return response


# Usage
if __name__ == '__main__':
    model = MiniCPMO26(model_id="openbmb/MiniCPM-o-2_6")

    response = model.run_inference(
        prompt="List the objects on the table.",
        image_path="/scratch/rsvargh2/occlusion_images/images/no/1.jpg"
    )
    print(response)
