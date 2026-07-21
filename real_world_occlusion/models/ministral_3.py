from unsloth import FastVisionModel # FastLanguageModel for LLMs

import torch
from PIL import Image
from transformers import TextStreamer

class Ministral3:
    #  unsloth/Ministral-3-8B-Instruct-2512
    def __init__(self, model_name="unsloth/Ministral-3-14B-Instruct-2512"):
        self.model, self.tokenizer = FastVisionModel.from_pretrained(
            model_name,
            load_in_4bit = False, # Use 4bit to reduce memory use. False for 16bit LoRA.
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
        
        text_streamer = TextStreamer(self.tokenizer, skip_prompt = True)
        generated_ids = self.model.generate(**inputs, max_new_tokens = 1000, use_cache = True, streamer = text_streamer,
                                               temperature = 1.5, min_p = 0.1)  
        # print(f"\nGen ids: {generated_ids}\n")

        generated_ids_trimmed = generated_ids[0][len(inputs.input_ids[0]):]
        # print(f"\nGen ids trimmed: {generated_ids_trimmed}\n")
        
        output_text = self.tokenizer.decode(generated_ids_trimmed, skip_special_tokens=True)

        return output_text, None, None

if __name__ == "__main__":
    model = Ministral3()

    response, _, _ = model.run_inference(
        prompt="List the objects on the table.",
        image_path="/scratch/rsvargh2/occlusion_images/exp2/images/no/1.jpg"
    )
    print(response)