from transformers import AutoModelForCausalLM, AutoTokenizer


class Qwen3Thinking:
    def __init__(self, model_name: str = "Qwen/Qwen3-4B-Instruct-2507", cache_dir="/scratch/rsvargh2/huggingface_models/"):
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype="auto",
            device_map="auto",
            cache_dir=cache_dir
        )

    def run_inference(self, prompt: str, max_new_tokens: int = 32768) -> dict:
        messages = [{"role": "user", "content": prompt}]

        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)

        generated_ids = self.model.generate(
            **model_inputs,
            max_new_tokens=max_new_tokens
        )
        output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist()

        # Parse thinking content
        try:
            # rindex finding 151668 (</think>)
            index = len(output_ids) - output_ids[::-1].index(151668)
        except ValueError:
            index = 0

        thinking_content = self.tokenizer.decode(output_ids[:index], skip_special_tokens=True).strip("\n")
        content = self.tokenizer.decode(output_ids[index:], skip_special_tokens=True).strip("\n")

        return {
            "thinking_content": thinking_content,
            "content": content
        }


if __name__ == "__main__":
    model = QwenThinkingModel()
    output = model.run_inference("Give me a short introduction to large language model.")
    print("Thinking content:", output["thinking_content"])
    print("Content:", output["content"])