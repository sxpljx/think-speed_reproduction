import re
import json

input_questions1 = "../datasets/train_12500.json"
input_questions2 = "./datasets/stimuli.json"

with open(input_questions1, "r", encoding="utf-8") as f:
    datas2 = json.load(f)

with open(input_questions2, "r", encoding="utf-8") as f:
    datas1 = json.load(f)

n = 0
correct = []
for data in datas:
    real_answer = data1["solution"]
    matches = re.findall(r'\\boxed\{([^}]*)\}', real_answer)
    answer_t = matches[-1] if matches else None

    model_answer_f = data2[n]["fast_response"]
    matches = re.findall(r'\\boxed\{([^}]*)\}', real_answer)
    answer_f = matches[-1] if matches else None

    model_answer_s = data2[n]["slow_response"]
    matches = re.findall(r'\\boxed\{([^}]*)\}', real_answer)
    answer_s = matches[-1] if matches else None

    #matches = re.findall(r'\\boxed\{([^}]*)\}', real_answer)
    #answer = matches[-1] if matches else None

    question = data2["question"]

    if answer_f == answer_t & answer_s == answer_t:
        correct.append({
            "user_prompt" : question,
            "fast_response" : model_answer_f,
            "slow_response" : model_answer_s,
            "answer" : answer_t
        })
    print(answer_t)
    n += 1
    if n > 5:
        break
