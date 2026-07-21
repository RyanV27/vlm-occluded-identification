from PIL import Image
from unsloth import FastModel
from transformers import TextStreamer
import torch

torch._dynamo.config.recompile_limit = 64

class Gemma3:
    # "unsloth/gemma-3n-E4B-it-unsloth-bnb-4bit"
    # "unsloth/gemma-3n-E4B-it"
    def __init__(self, model_name="unsloth/gemma-3-12b-it-unsloth-bnb-4bit"):
        self.model, self.tokenizer = FastModel.from_pretrained(
            model_name = model_name,
            dtype = None, # None for auto detection
            max_seq_length = 1024, # Choose any for long context!
            load_in_4bit = True,  # 4 bit quantization to reduce memory
            full_finetuning = False, # [NEW!] We have full finetuning now!
            # token = "YOUR_HF_TOKEN", # HF Token for gated models
        )

    def run_inference(self, prompt, image_path, max_new_tokens = 512):  
        image = Image.open(image_path)
        
        messages = [{
            "role" : "user",
            "content": [
                { "type": "image", "image" : image },
                { "type": "text",  "text" : prompt }
            ]
        }]

        with torch.no_grad():
            inputs = self.tokenizer.apply_chat_template(
                messages,
                add_generation_prompt = True, # Must add for generation
                tokenize = True,
                return_dict = True,
                return_tensors = "pt",
            ).to("cuda")
            # print(f"Input ids: {len(inputs.input_ids[0])}")
             
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens = max_new_tokens,
                temperature=0.0000001,
                # use_cache=False,
                # temperature = 1.0, top_p = 0.95, top_k = 64,
                # streamer = TextStreamer(self.tokenizer, skip_prompt = True),
            )
            # print(f"\nGen ids: {generated_ids}\n")      
        
            generated_ids_trimmed = generated_ids[0][len(inputs.input_ids[0]):]
            # print(f"\nGen ids trimmed: {generated_ids_trimmed}\n")
            
            output_text = self.tokenizer.decode(generated_ids_trimmed, skip_special_tokens=True)

            # _ = self.model.generate(
            #     **self.tokenizer.apply_chat_template(
            #         messages,
            #         add_generation_prompt = True, # Must add for generation
            #         tokenize = True,
            #         return_dict = True,
            #         return_tensors = "pt",
            #     ).to("cuda"),
            #     max_new_tokens = max_new_tokens,
            #     temperature = 1.0, top_p = 0.95, top_k = 64,
            #     streamer = TextStreamer(self.tokenizer, skip_prompt = True),
            # )
    
        return output_text, None, None

if __name__ == "__main__":
    model = Gemma3()

    response, _, _ = model.run_inference(
        prompt="List the objects on the table.",
        image_path="/scratch/rsvargh2/occlusion_images/exp2/images/no/1.jpg"
    )
    print(response)