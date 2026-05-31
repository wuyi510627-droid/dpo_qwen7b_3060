
import torch
from datasets import load_dataset

from transformers import TrainingArguments
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import DPOTrainer
from trl import DPOConfig  # 从 trl 导入，不是 transformers
print(f"CUDA: {torch.cuda.is_available()}, GPU: {torch.cuda.get_device_name(0)}")
print(f"显存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

def main():
    bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,              # 4-bit 加载
    bnb_4bit_use_double_quant=True, # 双量化，再省一点显存
    bnb_4bit_quant_type="nf4",      # NormalFloat 4-bit，QLoRA 推荐的量化类型
    bnb_4bit_compute_dtype=torch.float16  # 计算时用 bfloat16
)
    # 2. 加载模型
    model_name = "/home/wuyi/cuda12-dev/models/Qwen2.5-7B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_name, quantization_config=bnb_config, 
        device_map="auto", trust_remote_code=True
    )
    model.gradient_checkpointing_enable()
    
    # 3. LoRA 配置
    model = prepare_model_for_kbit_training(model)
    lora_config = LoraConfig(
        r=8, lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.05, bias="none", task_type="CAUSAL_LM"
    )
    model = get_peft_model(model, lora_config)
    
    # 4. 加载数据集
    # dataset = load_dataset("argilla/distilabel-intel-orca-dpo-pairs")
    # dataset = dataset.filter(lambda x: x["status"] != "tie" and x["chosen_score"] >= 8)
    # print(dataset[0])
    # def format_dpo_example(example):
    #     return {"prompt": example["input"],
    #     "chosen": example["chosen"],
    #     "rejected": example["rejected"]
    # }

    # dataset = dataset.map(format_dpo_example)
    # 4. 加载数据集（修正1：先取 train split）
    dataset_dict = load_dataset("argilla/distilabel-intel-orca-dpo-pairs")
    dataset = dataset_dict["train"]  # 关键！选择 train split
    
    # 过滤数据
    dataset = dataset.filter(lambda x: x["status"] != "tie" and x["chosen_score"] >= 8)
    
    print(f"数据集大小: {len(dataset)}")
    print(f"数据集列名: {dataset.column_names}")
    
    # 5. 格式化数据（修正2：使用正确的字段名）
    def format_dpo_example(example):
        return {
            "prompt": example["input"],           # 不是 input
            "chosen": example["chosen"],      # 不是 chosen
            "rejected": example["rejected"]   # 不是 rejected
        }
    
    dataset = dataset.map(format_dpo_example)
    print(f"格式化后列名: {dataset.column_names}")

    # 5. DPO 配置
    dpo_config = DPOConfig(
    output_dir="./dpo_output",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    num_train_epochs=1,
    learning_rate=5e-5,
    fp16=True,
    optim="adamw_8bit",
    logging_steps=10,
    save_steps=500,
    warmup_ratio=0.1,
    # DPO 特有参数可以放在这里，也可以放在 DPOTrainer 的参数中
)
    trainer = DPOTrainer(
    model=model,
    ref_model=None,
    args=dpo_config,  # 传入 DPOConfig 实例
    train_dataset=dataset,
    tokenizer=tokenizer,
    beta=0.1,         # 也可以在这里设置
    max_length=512,
)


    
    
    trainer.train()
    trainer.save_model("./dpo_final_model")

    
    


if __name__ == "__main__":
    main()
    

