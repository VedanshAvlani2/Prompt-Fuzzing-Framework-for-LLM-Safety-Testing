from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import torch, time

class LlamaAdapter:
    def __init__(self, model_name="meta-llama/Llama-2-7b-chat-hf"):
        self.model_name = model_name
        print(f"[llama_adapter] Loading model: {model_name}")

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        # bitsandbytes quantization config
        bnb_config = BitsAndBytesConfig(
            load_in_8bit=True,
            llm_int8_enable_fp32_cpu_offload=True,  # offload overflow weights to CPU
        )

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            device_map="auto",             # auto-balances layers across GPU/CPU
            low_cpu_mem_usage=True,
            dtype=torch.float16,
        )

        if torch.cuda.is_available():
            print(f"✅ Using GPU: {torch.cuda.get_device_name(0)}")

    def run(self, prompt: str):
        print(f"[{self.model_name}] starting inference…")
        start = time.time()

        inputs = self.tokenizer(prompt, return_tensors="pt").to("cuda")
        input_length = inputs.input_ids.shape[1]
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=200,
            temperature=0.7,
            do_sample=True,
            pad_token_id=self.tokenizer.eos_token_id,
        )

        generated_ids = outputs[0][input_length:]
        text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        if not text or text.strip() == prompt.strip():
            full_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            # Remove the prompt from the beginning
            if full_text.startswith(prompt):
                text = full_text[len(prompt):].strip()
            else:
                text = full_text
            
        elapsed = time.time() - start
        num_tokens = len(generated_ids)

        print(f"[{self.model_name}] finished inference in {elapsed:.1f}s.")
        print(f"[{self.model_name}] Generated {num_tokens} tokens")
        print(f"[{self.model_name}] Response preview: {text[:100]}...")

        return {
        "provider": "llama",
        "model": self.model_name,
        "text": text,
        "status": "ok",
        "latency_s": round(elapsed, 2),
        "num_tokens": num_tokens
        }
