# Paper reproduction
## Controlling Thinking Speed in Reasoning Models
## this product aim to reproduct a para named Controlling Thinking Speed in Reasoning Models (https://github.com/D2I-ai/thinking-speed-control) to start my study.
## 论文内容：
1. 通过分析大模型对于多个问题的回答，发现了大模型回答的首个词语会对大模型的回答token数量的多少有影响，所以论文认为通过限制大模型回答的首个token可以操控大模型回答的token数，即控制推理速度。
2. 通过1的结果可以让大模型对同一个问题生成快慢思考两种回答，然后对比中间隐藏状态，发现了一个慢思考到快思考的方向向量。
3. 通过这个方向向量，开发了一个系统可以通过分析难度控制思考速度
## 我的代码：
1. 如何使用可以参考论文原始项目仓库
2. 以下是我的代码介绍：
   count_token.py-计算输出token数
   get_answer_all.py-提取出大模型的回答
   get_hidden.py-得到中间隐藏状态
   get_initial_segment.py-得到推理链前几步
   judge_answer.py-判断回答正确与否
   new_wait.py-模型重复思考
   train_fast_to_slow.py-得到方向向量
