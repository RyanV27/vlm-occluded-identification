from transformers import AutoProcessor, AutoModelForImageTextToText
import torch
import requests
from PIL import Image


class Molmo2:
    def __init__(self, cache_dir="/scratch/rsvargh2/huggingface_models/"):
        self.model_id = "allenai/Molmo2-8B"
        self.processor = AutoProcessor.from_pretrained(
            self.model_id,
            trust_remote_code=True,
            dtype="auto",
            device_map="auto",
            cache_dir=cache_dir
        )
        self.model = AutoModelForImageTextToText.from_pretrained(
            self.model_id,
            trust_remote_code=True,
            dtype="auto",
            device_map="auto",
            cache_dir=cache_dir
        )

    @staticmethod
    def load_image(image_source):
        """Load an image from a file path, URL, or PIL Image."""
        if isinstance(image_source, Image.Image):
            return image_source
        elif isinstance(image_source, str) and image_source.startswith("http"):
            return Image.open(requests.get(image_source, stream=True).raw)
        else:
            return Image.open(image_source)

    def run_inference(self, prompt, image_path, max_new_tokens=448):
        """
        Run inference on the model.

        Args:
            prompt      (str): Text prompt.
            images      (PIL.Image | list): A single image or list of images.
                                            Each image can be a PIL Image, file path, or URL.
            max_new_tokens (int): Maximum number of tokens to generate.

        Returns:
            str: The generated text response.
        """
        if not isinstance(image_path, list):
            images = [image_path]
        images = [self.load_image(img) for img in images]

        content = [dict(type="text", text=prompt)]
        content += [dict(type="image", image=img) for img in images]

        messages = [{"role": "user", "content": content}]

        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
        )
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        with torch.inference_mode():
            generated_ids = self.model.generate(**inputs, max_new_tokens=max_new_tokens)

        generated_tokens = generated_ids[0, inputs['input_ids'].size(1):]

        output_texts = self.processor.tokenizer.decode(generated_tokens, skip_special_tokens=True)

        # 2nd and 3rd return values are generated ids and inputs ids from the model.
        # Can be added later if finding uncertainty.
        return output_texts, None, None


# Usage
if __name__ == '__main__':
    model = Molmo2(cache_dir="/scratch/rsvargh2/huggingface_models/")

    response = model.run_inference(
        prompt="Compare these images.",
        images=[
            "https://picsum.photos/id/237/536/354",
            "https://vllm-public-assets.s3.us-west-2.amazonaws.com/vision_model_images/cherry_blossom.jpg"
        ]
    )
    print(response)