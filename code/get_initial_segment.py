import json
import os
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

input_questions = "./datasets/stimuli.json"
output_initial_segment = './datasets/output_initial_segment.json'

def extract_initial_segment_by_steps(
    response: str,
    max_steps: int = 2,
    max_tokens: int = 256,
    ):
    """
    论文一致的 initial segment 提取：
    - <think> ... </think>
    - 用 \\n\\n 作为 step 分隔
    - 取前 max_steps
    - 再做 token-level safety 截断
    """

    # 1. 取 <think> 内内容
    if "<think>" in response:
        text = response.split("<think>", 1)[1]
    else:
        text = response

    if "</think>" in text:
        text = text.split("</think>", 1)[0]

    text = text.strip()

    # 2. 按 double newline 分 step
    steps = [s.strip() for s in text.split("\n\n") if s.strip()]
    for i in steps:
        print(i)
        print("#######")
    print("-----------------------------------")
    # 3. 取前 max_steps
    initial_steps = steps[:max_steps]
    initial_text = "\n\n".join(initial_steps)

    # 4. token-level 截断（防止异常长）
    #token_ids = tokenizer.encode(initial_text, add_special_tokens=False)
    #token_ids = token_ids[:max_tokens]
    #truncated = tokenizer.decode(token_ids)

    return initial_text

with open(input_questions, "r", encoding="utf-8") as f:
    datas = json.load(f)
initial_segment = []
for data in datas:
    question = data["question"]
    fast_response = extract_initial_segment_by_steps(data["fast_response"])
    slow_response = extract_initial_segment_by_steps(data["slow_response"])

    initial_segment.append({
        "question": question,
        "fast_response": fast_response,
        "slow_response": slow_response
    })
with open(output_initial_segment, "w", encoding="utf-8") as f:
    json.dump(initial_segment, f, ensure_ascii=False, indent=2)

