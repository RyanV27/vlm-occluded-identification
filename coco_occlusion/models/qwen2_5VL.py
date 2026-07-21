import base64
import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor, AutoModelForCausalLM
from qwen_vl_utils import process_vision_info

class Qwen25VL:
    def __init__(self, cache_dir="/scratch/rsvargh2/huggingface_models/"):
        # Initialize model on available device
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            "Qwen/Qwen2.5-VL-7B-Instruct",      # Earlier used "Qwen/Qwen2.5-VL-72B-Instruct", "Qwen/Qwen2.5-VL-72B-Instruct-AWQ"
            dtype=torch.float16,        # torch_dtype="auto", 
            device_map="auto",          # "cpu"
            cache_dir=cache_dir         # "/scratch/rsvargh2/huggingface_models/"
        )
        
        self.processor = AutoProcessor.from_pretrained(
           "Qwen/Qwen2.5-VL-7B-Instruct",     # Earlier used "Qwen/Qwen2.5-VL-72B-Instruct", "Qwen/Qwen2.5-VL-72B-Instruct-AWQ",
            use_fast=False,
            cache_dir=cache_dir,     # "/scratch/rsvargh2/huggingface_models/"
        )    

    def run_inference(self, prompt, image_path, num_beams=1, num_return_sequences=1,
                        temperature=0.000001, top_p=1.0, do_sample=False, num_generations=1, output_logits=False):
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": "file://" + image_path,
                    },
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        
        # Preparation for inference
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to("cuda")
        
        # Inference: Generation of the output
        output_texts = []
        generations_generated_ids = []
        for _ in range(num_generations):
            if output_logits == True:
                print("Output with logits!")
                model_output = self.model.generate(
                    **inputs, 
                    max_new_tokens=512,
                    num_beams=num_beams,
                    num_return_sequences=num_return_sequences,
                    temperature=temperature, 
                    top_p=top_p,
                    do_sample=do_sample,
                    logits_to_keep=5,
                    output_logits=output_logits,
                    return_dict_in_generate=True,
                )
                generated_ids = model_output['sequences']
                print(f"Model output logits: {model_output['logits'][0]}")
                print(f"Model output logits shape: {model_output['logits'][0].shape}")
            else:
                generated_ids = self.model.generate(
                    **inputs, 
                    max_new_tokens=512,
                    num_beams=num_beams,
                    num_return_sequences=num_return_sequences,
                    temperature=temperature, 
                    top_p=top_p,
                    do_sample=do_sample,
                )
            
            generations_generated_ids.append(generated_ids)
            
            generated_ids_trimmed = [
                out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            output_text = self.processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )
    
            output_texts.append(output_text)
    
        return output_texts[0][0], generations_generated_ids, inputs.input_ids