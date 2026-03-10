import json
import os
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

# =========================
# 路径配置
# =========================

model_path = "../../../../../models/LRM/DeepSeek-R1-Distill-Qwen-7B"

input_questions = "../datasets/train_12500.json"
output_stimuli = "./datasets/all_stimuli_final.json"

os.makedirs(os.path.dirname(output_stimuli), exist_ok=True)

device = "cuda" if torch.cuda.is_available() else "cpu"


# =========================
# 加载模型
# =========================

tokenizer = AutoTokenizer.from_pretrained(
    model_path,
    trust_remote_code=True
)

model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.float16,
    device_map="auto",
    trust_remote_code=True
)
model.eval()


# =========================
# 生成函数
# =========================

def generate(prompt, max_new_tokens):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.6,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id
        )

    return tokenizer.decode(outputs[0], skip_special_tokens=True)


# =========================
# 读取问题
# =========================

with open(input_questions, "r", encoding="utf-8") as f:
    questions = json.load(f)

print(f"Total questions: {len(questions)}")


# =========================
# 主循环
# =========================

stimuli = []
n = 0
for idx, item in enumerate(tqdm(questions, desc="Generating fast/slow stimuli", disable=True)):
    question = item["user_prompt"]

    # ---------- fast-thinking（你建议的方式） ----------
    fast_prompt = f"""<|User|>
{question}
<|Assistant|>
<think>
To"""

    # ---------- slow-thinking ----------
    slow_prompt = f"""<|User|>
{question}
<|Assistant|>
<think>
"""

    fast_response = generate(
        prompt=fast_prompt,
        max_new_tokens=32768   # fast 也允许完整生成，只是自然会更短
    )

    slow_response = generate(
        prompt=slow_prompt,
        max_new_tokens=32768
    )

    stimuli.append({
        "user_prompt": question,
        "fast_response": fast_response,
        "slow_response": slow_response
    })

    # 防中断保存
    if (idx + 1) % 10 == 0:
        with open(output_stimuli, "w", encoding="utf-8") as f:
            json.dump(stimuli, f, ensure_ascii=False, indent=2)
    n += 1
    if n % 50 == 0:
        print(f'完成推理question{n}')
    if n > 7500:
        break


# =========================
# 最终保存
# =========================

with open(output_stimuli, "w", encoding="utf-8") as f:
    json.dump(stimuli, f, ensure_ascii=False, indent=2)

print(f"Saved stimuli to {output_stimuli}")
