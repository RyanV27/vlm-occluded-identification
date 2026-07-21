import math
import numpy as np
import torch
import torchvision.transforms as T
from decord import VideoReader, cpu
from PIL import Image
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoModel, AutoTokenizer

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class InternVL35:
    def __init__(self, cache_dir="/scratch/rsvargh2/huggingface_models/"):
        self.hugging_face_path = 'OpenGVLab/InternVL3_5-8B'
        self.model = AutoModel.from_pretrained(
            self.hugging_face_path,
            dtype=torch.bfloat16,
            # load_in_8bit=False,
            low_cpu_mem_usage=True,   # 
            # use_flash_attn=True,
            trust_remote_code=True,
            device_map="auto",     # "auto"
            cache_dir=cache_dir
        ).eval()
        # self.model = self.model.to("cuda")
        self.tokenizer = AutoTokenizer.from_pretrained(self.hugging_face_path, trust_remote_code=True, use_fast=False, cache_dir=cache_dir)
        # self.tokenizer = self.tokenizer.to("cuda")

    def build_transform(self, input_size):
        transform = T.Compose([
            T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
            T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
        ])
        return transform

    def find_closest_aspect_ratio(self, aspect_ratio, target_ratios, width, height, image_size):
        best_ratio_diff = float('inf')
        best_ratio = (1, 1)
        area = width * height
        for ratio in target_ratios:
            target_aspect_ratio = ratio[0] / ratio[1]
            ratio_diff = abs(aspect_ratio - target_aspect_ratio)
            if ratio_diff < best_ratio_diff:
                best_ratio_diff = ratio_diff
                best_ratio = ratio
            elif ratio_diff == best_ratio_diff:
                if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                    best_ratio = ratio
        return best_ratio

    def dynamic_preprocess(self, image, min_num=1, max_num=12, image_size=448, use_thumbnail=False):
        orig_width, orig_height = image.size
        aspect_ratio = orig_width / orig_height

        target_ratios = set(
            (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1)
            if i * j <= max_num and i * j >= min_num
        )
        target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

        target_aspect_ratio = self.find_closest_aspect_ratio(
            aspect_ratio, target_ratios, orig_width, orig_height, image_size)

        target_width = image_size * target_aspect_ratio[0]
        target_height = image_size * target_aspect_ratio[1]
        blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

        resized_img = image.resize((target_width, target_height))
        processed_images = []
        for i in range(blocks):
            box = (
                (i % (target_width // image_size)) * image_size,
                (i // (target_width // image_size)) * image_size,
                ((i % (target_width // image_size)) + 1) * image_size,
                ((i // (target_width // image_size)) + 1) * image_size
            )
            processed_images.append(resized_img.crop(box))
        assert len(processed_images) == blocks
        if use_thumbnail and len(processed_images) != 1:
            processed_images.append(image.resize((image_size, image_size)))
        return processed_images

    def load_image(self, image_file, input_size=448, max_num=12):
        image = Image.open(image_file).convert('RGB')
        transform = self.build_transform(input_size=input_size)
        images = self.dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
        pixel_values = [transform(img) for img in images]
        return torch.stack(pixel_values)

    def run_inference(self, prompt, image_path, max_num=12, max_new_tokens=1024, do_sample=True):
        pixel_values = self.load_image(image_path, max_num=max_num).to(torch.bfloat16).cuda()
        generation_config = dict(max_new_tokens=max_new_tokens, do_sample=do_sample)
        question = f'<image>\n{prompt}'
        response = self.model.chat(self.tokenizer, pixel_values, question, generation_config)
        # 2nd and 3rd are generated ids and inputs ids from the model.
        # Can be added later if finding uncertainty.
        return response, None, None     


# Usage
if __name__ == '__main__':
    model = InternVL35(path='OpenGVLab/InternVL3_5-8B')
    response = model.run_inference(
        prompt='Please describe the image shortly.',
        image_file='/scratch/rsvargh2/occlusion_images/images/no/1.jpg'   # './examples/image1.jpg'
    )
    print(f'Assistant: {response}')