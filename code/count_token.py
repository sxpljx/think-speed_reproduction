from transformers import AutoTokenizer
import random
import re
import json

# 加载你存放模型本地的路径
model_path = "../../../../../models/LRM/DeepSeek-R1-Distill-Qwen-7B"
#answer_path = "../result/wait_outputs/aime24/wait_3.json"
#answer_path = "../result/deepseek_7B_mt115031_aime24_8"
answer_path = '../result/deepseek_7B_to_aime24_8'
real_path = "../datasets/aime24.json"
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

# 需要统计的模型回答文本
with open(answer_path, "r", encoding="utf-8") as f1:
    datas1 = json.load(f1)
with open(real_path, "r", encoding="utf-8") as f2:
    datas2 = json.load(f2)

n = 0
count = 0
total_token = 0
count_all = 0

for i in range(10):
    for data in datas1:
        r = random.randint(0, 7)
        model_answer = data["model_response"][r]
        matches = re.findall(r'\\boxed\{([^}]*)\}', model_answer)
        print("r:", r)
        print('match:', matches)
        
        if len(matches) > 0:
            answer_m = matches[-1] if matches else None
            answer_m = answer_m
            real_answer = data["answer"]
            print(real_answer)
            real_answer = real_answer
            if answer_m == real_answer:
                count += 1
        else:
            if r == 7:
                r = r - 1
            else:
                r += 1
            print('====================================')
            model_answer = data["model_response"][r]
            matches = re.findall(r'\\boxed\{([^}]*)\}', model_answer)
            answer_m = matches[-1] if matches else None
            answer_m = answer_m
            real_answer = data["answer"]
            print(matches)
            print(real_answer)
            real_answer = real_answer
            print('====================================')
            if answer_m == real_answer:
                count += 1
        
        n += 1
        print("正确数:",count)
        print('第几个:',n)
        token_count = 0   
        for s in data["model_response"]:
            encoded_ids = tokenizer.encode(s)
            # 计算Token数量
            token_count += len(encoded_ids)
        
        token_count = int(token_count / 8)
        print(f"回答的平均Token数量为: {token_count}")
        total_token += token_count
    #print(ss)
    #print(i)
    count_all += (count/n)

'''
#真实值和模型回答不在同一个文件里
for i in range(10):
    n = 0
    count = 0
    for data in datas1:
        r = random.randint(0, 7)
        model_answer = data["model_response"][r]
        matches = re.findall(r'\\boxed\{([^}]*)\}', model_answer)
        #print("r:", r)
        #print('match:', matches)
        if len(matches) > 0:
            answer_m = matches[-1] if matches else None
            answer_m = answer_m
            real_answer = datas2[n]["answer"]
            real_answer = real_answer
            if answer_m == real_answer:
                count += 1
        else:
            if r == 7:
                r -= 1
            else:
                r += 1
            model_answer = data["model_response"][r]
            matches = re.findall(r'\\boxed\{([^}]*)\}', model_answer)
            answer_m = matches[-1] if matches else None
            answer_m = answer_m
            real_answer = datas2[n]["answer"]
            real_answer = real_answer
            if answer_m == real_answer:
                count += 1
        n += 1
        #print("正确数:",count)
        #print('第几个:',n)
        token_count = 0
        for s in data["model_response"]:
            encoded_ids = tokenizer.encode(s)
            # 计算Token数量
            token_count += len(encoded_ids)
            
            
            #text = tokenizer.decode(s,skip_special_tokens=True,clean_up_tokenization_spaces=True)
            #token_count_s = len(tokenizer.encode(text, add_special_tokens=False))
            
            #token_count += token_count_s
            
        token_count = int(token_count / 8)
        #print(f"回答的平均Token数量为: {token_count}")
        total_token += token_count
    print(i)
    print(n)
    print(count/n)
    count_all += (count/n)
'''
print(n)
print(count_all/10)
print(total_token/n)




