# DPO 微调 Qwen2.5-7B（RTX 3060 12G 适配）

基于 Direct Preference Optimization（DPO）算法对 Qwen2.5-7B-Instruct 进行微调，针对 RTX 3060 12GB 显存做了专项优化，通过 4-bit 量化 + LoRA 实现消费级显卡上的大模型微调与推理。

---

## 文件说明

| 文件 | 说明 |
|------|------|
| `train.py` | DPO 微调训练脚本 |
| `infer.py` | 加载微调后模型的交互式推理脚本 |
| `orl_infer.py` | 原始 Qwen2.5-7B-Instruct 推理脚本（支持量化/全精度两种模式） |

---

## 环境要求

- GPU：RTX 3060 12GB 或同等显存
- CUDA：11.8+
- Python：3.10+

安装依赖：

```bash
pip install torch transformers peft trl datasets bitsandbytes
```

---

## 使用方式

### 1. 训练（DPO 微调）

```bash
python train.py
```

训练配置：
- 基础模型：Qwen2.5-7B-Instruct
- 数据集：Orca DPO Pairs
- 量化：4-bit NormalFloat + 双量化
- LoRA：rank=8，目标模块为注意力头（q/k/v/o_proj）
- 批大小：1，梯度累积步数：8，学习率：5e-5
- 微调权重保存至 `./dpo_final_model`

### 2. 推理（微调后模型）

```bash
python infer.py
```

加载 `./dpo_final_model` 中的 LoRA 适配器，进入交互式对话，输入 `quit` / `exit` / `q` 退出。

### 3. 推理（原始模型）

```bash
python orl_infer.py
```

直接加载原始 Qwen2.5-7B-Instruct，支持量化和全精度两种模式，以及批量生成（取消注释 `batch_generate_example()` 即可）。

---

## 核心优化技术

- **4-bit 量化**（bitsandbytes）：显存占用降低约 75%
- **LoRA 微调**（peft）：仅训练少量参数，不修改原模型权重
- **梯度检查点**：以计算时间换显存空间
- **梯度累积**：小批量模拟大批量训练效果

---

## 生成参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| max_new_tokens | 512 | 最大生成 token 数 |
| temperature | 0.7 | 生成多样性，0 为确定性输出 |
| top_p | 0.9 | nucleus 采样阈值 |
| repetition_penalty | 1.05 | 重复惩罚系数 |
