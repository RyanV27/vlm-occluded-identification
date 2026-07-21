from transformers import AutoModelForCausalLM, AutoProcessor, GenerationConfig
from PIL import Image
import requests


class Molmo:
    def __init__(self, cache_dir="/scratch/rsvargh2/huggingface_models/"):
        self.model_id = 'allenai/Molmo-7B-D-0924'
        kwargs = dict(trust_remote_code=True, torch_dtype='auto', device_map='auto')
        if cache_dir:
            kwargs['cache_dir'] = cache_dir

        self.processor = AutoProcessor.from_pretrained(self.model_id, **kwargs)
        self.model = AutoModelForCausalLM.from_pretrained(self.model_id, **kwargs)

    @staticmethod
    def load_image(image_source):
        """Load an image from a file path, URL, or PIL Image."""
        if isinstance(image_source, Image.Image):
            return image_source
        elif isinstance(image_source, str) and image_source.startswith("http"):
            return Image.open(requests.get(image_source, stream=True).raw)
        else:
            return Image.open(image_source)

    def run_inference(self, prompt, image_path, max_new_tokens=200):
        """
        Run inference on the model.

        Args:
            prompt         (str): Text prompt.
            image_path (PIL.Image | str | list): A single image or list of images.
                                                 Each image can be a PIL Image, file path, or URL.
            max_new_tokens (int): Maximum number of tokens to generate.

        Returns:
            str: The generated text response.
        """
        if not isinstance(image_path, list):
            image_path = [image_path]
        images = [self.load_image(img) for img in image_path]

        inputs = self.processor.process(images=images, text=prompt)
        inputs = {k: v.to(self.model.device).unsqueeze(0) for k, v in inputs.items()}

        output = self.model.generate_from_batch(
            inputs,
            GenerationConfig(max_new_tokens=max_new_tokens, stop_strings="<|endoftext|>"),
            tokenizer=self.processor.tokenizer
        )

        generated_tokens = output[0, inputs['input_ids'].size(1):]
        output_text = self.processor.tokenizer.decode(generated_tokens, skip_special_tokens=True)

        return output_text, None, None


# Usage
if __name__ == '__main__':
    model = Molmo(cache_dir="/scratch/rsvargh2/huggingface_models/")

    response, _, _ = model.run_inference(
        prompt="Describe this image.",
        image_path="/scratch/rsvargh2/occlusion_images/images/1.jpg"
    )
    print(response)