import json
import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# =========================
# 路径配置
# =========================
model_path = "../../../../models/LRM/DeepSeek-R1-Distill-Qwen-7B"
input_json = "./datasets/train_12500.json"
output_dir = "./result/wait_outputs/1"

os.makedirs(output_dir, exist_ok=True)

# =========================
# 读取所有 prompt
# =========================
with open(input_json, "r", encoding="utf-8") as f:
    data = json.load(f)

print(f"Total prompts: {len(data)}")

# =========================
# 加载 tokenizer & model
# =========================
tokenizer = AutoTokenizer.from_pretrained(
    model_path,
    trust_remote_code=True
)

model = AutoModelForCausalLM.from_pretrained(
    model_path,
    device_map="auto",
    torch_dtype=torch.float16,
    trust_remote_code=True
)
model.eval()

# =========================
# 一次生成 4 个回答（等价 vLLM t.text）
# =========================
def generate_4(prompt, max_new_tokens=32768):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    input_len = inputs["input_ids"].shape[1]

    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        temperature=0.6,
        top_p=0.9,
        do_sample=True,
        num_return_sequences=4   # ⭐ 一次生成 4 个
    )

    # 只保留新生成 token（不含 prompt）
    gen_ids = outputs[:, input_len:]

    # decode 成 4 个字符串（不 strip，不删 think）
    texts = tokenizer.batch_decode(
        gen_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=True
    )

    return texts  # List[str], len == 4

# =========================
# 为 0/1/2/3 Wait 准备容器
# =========================
all_results = {
    0: [],
    1: [],
    2: [],
    3: []
}

# =========================
# 主循环
# =========================
n = 0
for task_idx, task in enumerate(data):
    user_prompt = task["user_prompt"]

    print(f"\nRunning prompt {task_idx}")

    base_prompt = f"""<|User|>
{user_prompt}
<|Assistant|>
<think>
"""

    # ---------- 0 × Wait ----------
    responses = generate_4(base_prompt)

    # ⚠️ 保存方式 == vLLM 的 [t.text, t.text, t.text, t.text]
    all_results[0].append({
        "user_prompt": user_prompt,
        "model_response": responses
    })

    n += 1
    if n > 10:
        break

    # Wait 使用第一条完整输出（逻辑保持）
    current_prompt = responses[0]

    
    # ---------- 1 / 2 / 3 × Wait ----------
    for wait_idx in range(1, 4):
        next_prompt = current_prompt + "\nWait\n"

        responses = generate_4(next_prompt)

        all_results[wait_idx].append({
            "user_prompt": user_prompt,
            "model_response": responses
        })

        current_prompt = responses[0]
    break
    

# =========================
# 写文件
# =========================
for wait_idx in range(4):
    out_path = os.path.join(output_dir, f"wait_{wait_idx}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(all_results[wait_idx], f, ensure_ascii=False, indent=2)

print("\n完成：一次生成 4 个回答，保存方式等价 vLLM")
