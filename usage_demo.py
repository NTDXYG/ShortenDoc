from ShortenDoc import ShortenDoc
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

tokenizer = AutoTokenizer.from_pretrained("D:\models\CodeGPT-small-py-adaptedGPT2")
model = AutoModelForCausalLM.from_pretrained("D:\models\CodeGPT-small-py-adaptedGPT2",
            torch_dtype=torch.bfloat16)
SD = ShortenDoc(model, tokenizer, threshold=0.99, topk=5)

prompt = '''def tri(n):
    """Everyone knows Fibonacci sequence, it was studied deeply by mathematicians in 
    the last couple centuries. However, what people don't know is Tribonacci sequence.
    Tribonacci sequence is defined by the recurrence:
    tri(1) = 3
    tri(n) = 1 + n / 2, if n is even.
    tri(n) =  tri(n - 1) + tri(n - 2) + tri(n + 1), if n is odd.
    For example:
    tri(2) = 1 + (2 / 2) = 2
    tri(4) = 3
    tri(3) = tri(2) + tri(1) + tri(4)
           = 2 + 3 + 3 = 8 
    You are given a non-negative integer number n, you have to a return a list of the 
    first n + 1 numbers of the Tribonacci sequence.
    Examples:
    tri(3) = [1, 3, 2, 8]
    """
'''

prompt_list = [p for p in prompt.strip().split('\"\"\"') if len(p) > 0]
template = '\"\"\"'.join(prompt_list[:-1]) + '\"\"\"\n    '
text = prompt_list[-1].strip()


# 进行多次推理测试
import time
num_trials = 10
total_time = 0
for _ in range(num_trials):
    start_time = time.time()
    results, ratio = SD.shorten(text, template)
    end_time = time.time()
    total_time += end_time - start_time

# 计算平均推理时间
average_time = total_time / num_trials
print(f"平均推理时长: {average_time:.2f} 秒")

new_prompt = template + results[-1] + '\n    \"\"\"\n'
print(new_prompt)
print('Shorten Radio: ', ratio)