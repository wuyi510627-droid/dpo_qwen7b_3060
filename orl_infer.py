import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

BASE_MODEL = "/home/wuyi/cuda12-dev/models/Qwen2.5-7B-Instruct"

def load_model():
    """加载原模型（4-bit 量化）"""
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )
    
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )
    
    model.eval()
    return model, tokenizer


def load_model_full_precision():
    """加载原模型（全精度，不量化）- 需要更多显存"""
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.float16,  # 使用 float16 节省显存
    )
    
    model.eval()
    return model, tokenizer


def generate(model, tokenizer, prompt: str, max_new_tokens: int = 512, temperature: float = 0.7) -> str:
    """生成回复"""
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
            top_p=0.9,           #  nucleus sampling
            repetition_penalty=1.05,  # 轻微减少重复
        )

    new_tokens = output_ids[0][inputs["input_ids"].shape[-1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True)


def generate_single(model, tokenizer, prompt: str, max_new_tokens: int = 512) -> str:
    """确定性生成（temperature=0，结果稳定可复现）"""
    return generate(model, tokenizer, prompt, max_new_tokens, temperature=0.0)


def main():
    print("=" * 50)
    print("加载原模型 Qwen2.5-7B-Instruct...")
    print("=" * 50)
    
    model, tokenizer = load_model()
    
    print("模型加载完成！输入 'quit' 退出，输入 'clear' 清屏\n")
    print("-" * 50)

    while True:
        prompt = input("🧑 用户: ").strip()
        
        if prompt.lower() in ("quit", "exit", "q"):
            print("👋 再见！")
            break
        
        if prompt.lower() == "clear":
            print("\033[2J\033[H")  # 清屏
            continue
            
        if not prompt:
            continue
            
        response = generate(model, tokenizer, prompt)
        print(f"🤖 助手: {response}\n")
        print("-" * 50)


def batch_generate_example():
    """批量生成示例（无需交互）"""
    model, tokenizer = load_model()
    
    prompts = [
        "什么是深度学习？",
        "用一句话解释什么是 Transformer 架构",
        "Python 和 C++ 的主要区别是什么？",
    ]
    
    for prompt in prompts:
        print(f"\n问题: {prompt}")
        response = generate(model, tokenizer, prompt, max_new_tokens=200)
        print(f"回答: {response}")
        print("-" * 40)


if __name__ == "__main__":
    main()
    # 如果想用批量模式，注释上面，取消下面注释
    # batch_generate_example()