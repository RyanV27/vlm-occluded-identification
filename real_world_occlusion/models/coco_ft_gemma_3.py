from unsloth import FastVisionModel # FastLanguageModel for LLMs

import torch
from PIL import Image
from transformers import TextStreamer

class COCOFTGemma3:
    def __init__(self, model_save_path="/scratch/rsvargh2/finetuned_huggingface_models/unsloth/coco/gemma-3-12b-it-sft-lora-1"):
        self.model, self.tokenizer = FastVisionModel.from_pretrained(
            model_save_path,
            load_in_4bit = True, # Use 4bit to reduce memory use. False for 16bit LoRA.
            use_gradient_checkpointing = "unsloth", # True or "unsloth" for long context
        )

    def run_inference(self, prompt, image_path):
        FastVisionModel.for_inference(self.model) # Enable for inference!

        image = Image.open(image_path)

        messages = [
            {"role": "user", "content": [
                {"type": "image"},
                {"type": "text", "text": prompt}
            ]}
        ]
        input_text = self.tokenizer.apply_chat_template(messages, add_generation_prompt = True)
        inputs = self.tokenizer(
            image,
            input_text,
            add_special_tokens = False,
            return_tensors = "pt",
        ).to("cuda")
        # print(f"Input ids: {len(inputs.input_ids[0])}")
        
        # text_streamer = TextStreamer(tokenizer, skip_prompt = True)
        generated_ids = self.model.generate(**inputs, max_new_tokens = 512, use_cache = True)  # streamer = text_streamer,
        # print(f"\nGen ids: {generated_ids}\n")

        generated_ids_trimmed = generated_ids[0][len(inputs.input_ids[0]):]
        # print(f"\nGen ids trimmed: {generated_ids_trimmed}\n")
        
        output_text = self.tokenizer.decode(generated_ids_trimmed, skip_special_tokens=True)

        return output_text, None, None

if __name__ == "__main__":
    model = FTGemma3()

    response, _, _ = model.run_inference(
        prompt="List everything that is visible in the image.",
        image_path="/scratch/rsvargh2/synthetic_occlusion_images/coco/object/low/000000006040.jpg"
    )
    print(response)