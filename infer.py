import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

BASE_MODEL = "/home/wuyi/cuda12-dev/models/Qwen2.5-7B-Instruct"
ADAPTER_PATH = "./dpo_final_model"

def load_model(adapter_path: str = ADAPTER_PATH):
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )
    model = PeftModel.from_pretrained(base_model, adapter_path)
    model.eval()
    return model, tokenizer


def generate(model, tokenizer, prompt: str, max_new_tokens: int = 512, temperature: float = 0.7) -> str:
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=temperature > 0,
            pad_token_id=tokenizer.eos_token_id,
        )

    new_tokens = output_ids[0][inputs["input_ids"].shape[-1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True)


def main():
    print("Loading model...")
    model, tokenizer = load_model()
    print("Model loaded. Type 'quit' to exit.\n")

    while True:
        prompt = input("User: ").strip()
        if prompt.lower() in ("quit", "exit", "q"):
            break
        if not prompt:
            continue
        response = generate(model, tokenizer, prompt)
        print(f"Assistant: {response}\n")


if __name__ == "__main__":
    main()
