import json
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

# =========================
# 路径配置
# =========================

model_path = "../../../../../models/LRM/DeepSeek-R1-Distill-Qwen-7B"

# 输入：包含 fast / slow response 的 stimuli 文件
# 每个样本格式示例：
# {
#   "question": "...",
#   "fast_response": "...",
#   "slow_response": "..."
# }
input_json = "./datasets/output_initial_segment.json"

output_path = "./datasets/hidden_states.pt"

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
# 工具函数
# =========================

def extract_initial_segment(text: str, max_steps: int = 2):
    """
    提取前 max_steps 个 reasoning step
    假设 reasoning step 以换行或编号分隔
    """
    lines = text.strip().split("\n")
    return "\n".join(lines[:max_steps])


def get_last_token_hidden(prompt: str):
    """
    返回：
      hidden_states: list[num_layers] of Tensor[hidden_dim]
    """
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model(
            **inputs,
            output_hidden_states=True,
            return_dict=True
        )

    # hidden_states: tuple(num_layers + 1, B, T, D)
    hidden_states = outputs.hidden_states

    # 取最后一个 token
    last_token_idx = inputs["input_ids"].shape[1] - 1

    layer_hiddens = []
    for layer_idx in range(1, len(hidden_states)):  # 跳过 embedding 层
        h = hidden_states[layer_idx][0, last_token_idx]
        layer_hiddens.append(h.detach().cpu())

    return layer_hiddens


# =========================
# 主逻辑
# =========================

with open(input_json, "r", encoding="utf-8") as f:
    data = json.load(f)

num_samples = len(data)
print(f"Total stimuli pairs: {num_samples}")

h_fast_all = []
h_slow_all = []

for item in tqdm(data, desc="Extracting hidden states"):
    question = item["question"]
    fast_resp = item["fast_response"]
    slow_resp = item["slow_response"]

    # ---------- initial segment ----------
    #fast_init = extract_initial_segment(fast_resp, max_steps=2)
    #slow_init = extract_initial_segment(slow_resp, max_steps=2)

    # ---------- 构造模型输入 ----------
    fast_prompt = f"""<|User|>
{question}
<|Assistant|>
<think>
{fast_resp}
"""

    slow_prompt = f"""<|User|>
{question}
<|Assistant|>
<think>
{slow_resp}
"""

    # ---------- 提取隐藏状态 ----------
    h_fast = get_last_token_hidden(fast_prompt)
    h_slow = get_last_token_hidden(slow_prompt)

    h_fast_all.append(torch.stack(h_fast))  # [num_layers, hidden_dim]
    h_slow_all.append(torch.stack(h_slow))


# =========================
# 堆叠 & 保存
# =========================

# [num_samples, num_layers, hidden_dim] → [num_layers, num_samples, hidden_dim]
h_fast_tensor = torch.stack(h_fast_all).transpose(0, 1)
h_slow_tensor = torch.stack(h_slow_all).transpose(0, 1)

save_obj = {
    "h_fast": h_fast_tensor,
    "h_slow": h_slow_tensor,
    "meta": {
        "num_layers": h_fast_tensor.shape[0],
        "num_samples": h_fast_tensor.shape[1],
        "hidden_dim": h_fast_tensor.shape[2],
        "model": "DeepSeek-R1-Distill-Qwen-7B",
        "token_position": "last_token_of_initial_segment",
        "initial_segment": "first_2_reasoning_steps"
    }
}

torch.save(save_obj, output_path)
print(f"Saved hidden states to {output_path}")
