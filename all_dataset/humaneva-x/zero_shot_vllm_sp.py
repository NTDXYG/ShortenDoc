import os
import pandas as pd
from nlp2 import set_seed
from tqdm import tqdm
from human_eval.data import write_jsonl, read_problems
import torch
from vllm import LLM, SamplingParams
import os
from human_eval.evaluate_functional_correctness import entry_point
os.environ["TOKENIZERS_PARALLELISM"] = "false"
set_seed(42)

def main(model):
    torch.cuda.empty_cache()

    if model == 'ds-1.3':
        model_path = '/media/yg/E/models/deepseek-coder-1.3b-instruct'
    if model == 'ds-6.7':
        model_path = '/media/yg/E/models/deepseek-coder-6.7b-instruct'
    if model == 'codeqwen':
        model_path = '/media/yg/E/models/CodeQwen1.5-7B-Chat'
    if model == 'codegeex4':
        model_path = '/media/yg/E/models/codegeex4-all-9b'
    if model == 'llama3.1':
        model_path = '/media/yg/E/models/Meta-Llama-3.1-8B-Instruct'
        
    sampling_params = SamplingParams(temperature=0, max_tokens=512)

    llm = LLM(model=model_path, trust_remote_code=True, max_model_len=2048)
    
    # for method in ['random', 'selective_context', 'lingua', 'shorten_prompt']:
    for method in ['SC', 'lingua']:
        for dataset in ['cpp', 'go', 'java', 'js']:
        # for dataset in ['js']:
            problems = read_problems('./humaneva-x/'+dataset+'_humaneval.jsonl')
            if method == 'random':
                df = pd.read_json("./dataset/compare_study/Random_HumanEval_0.3.jsonl", lines=True)
                prompts = df['prompt'].tolist()
                reduces = []
                for prompt in prompts:
                    prompt = prompt.strip()
                    prompt_list = [p for p in prompt.split('\"\"\"') if len(p) > 0]
                    reduces.append(prompt_list[-1].strip())
            else:
                df = pd.read_json("./EM/"+method+"_HumanEval_0.3.jsonl", lines=True)
                reduces = df['reduced'].tolist()
            # print(reduces)
            samples = []
            task_ids = []
            codes = []
            idx = 0
            for task_id in tqdm(problems):
                prompt = problems[task_id]["prompt"]
                if dataset == 'cpp':
                    import re
                    try:
                        comment = re.findall(r'\/*.*?\*/', prompt, re.DOTALL)[-1]
                        prompt = prompt.replace(comment, "<COMMENT>")
                    except:
                        comment = re.findall(r'//.*', prompt, re.DOTALL)[-1]
                        prompt = prompt.replace(comment, "<COMMENT>")
                    prompt = prompt.replace("<COMMENT>", '/* ' + reduces[idx] + ' */')
                if dataset == 'go':
                    import re
                    if prompt.count("\nfunc ") > 1:
                        new_prompt = prompt[prompt.find("func"):]
                        comments = re.findall(r'//.*', new_prompt)
                    else:
                        comments = re.findall(r'//.*', prompt)
                    all_comments = '\n'.join(comments)
                    prompt = prompt.replace(all_comments, "<COMMENT>")
                    reduce = reduces[idx].replace('\n', '\n// ')
                    prompt = prompt.replace("<COMMENT>", '// ' + reduce)
                if dataset == 'java':
                    import re
                    comment = re.findall(r'/\*\*.*?\*/', prompt, re.DOTALL)[-1]
                    prompt = prompt.replace(comment, "<COMMENT>")
                    prompt = prompt.replace("<COMMENT>", '/**\n    ' + reduces[idx] + '\n     */')
                if dataset == 'js':
                    import re
                    comment = re.findall(r'\/*.*?\*/', prompt, re.DOTALL)[-1]
                    prompt = prompt.replace(comment, "<COMMENT>")
                    prompt = prompt.replace("<COMMENT>", '/* ' + reduces[idx] + ' */')
                # print(prompt)
                codes.append(prompt.strip() + '\n')
                task_ids.append(task_id)
                idx += 1
            outputs = llm.generate(codes, sampling_params)
            for i in range(len(outputs)):
                output = outputs[i]
                task_id = task_ids[i]
                text = output.outputs[0].text
                if dataset.lower() == 'java':
                    STOP_SEQS = ['public static void main', '[/code]', 'public class Main', '```', '[/java]', '\n###PATH']
                    for stop_seq in STOP_SEQS:
                        index = text.find(stop_seq)
                        if index != -1:
                            text = text[:index]
                    if text.count('{') + 1 == text.count('}'):
                        text += '\n}'
                elif dataset.lower() == 'go':
                    STOP_SEQS = ['func main', 'package main', '[/code]', '\n// ', '```', '[/go]', '\n###PATH']
                    for stop_seq in STOP_SEQS:
                        index = text.find(stop_seq)
                        if index != -1:
                            text = text[:index]
                elif dataset.lower() == 'cpp':
                    STOP_SEQS = ['\n/*\n', '\n/* ', 'int main', '```', '[/cpp]', '[/code]', '\n###PATH']
                    for stop_seq in STOP_SEQS:
                        index = text.find(stop_seq)
                        if index != -1:
                            text = text[:index]
                elif dataset.lower() == 'js':
                    STOP_SEQS = ['\n###PATH', '\n\nconsole', '\n\n/* ', '```', ';\n\n//', ';\n\n/*', '\n\nmodule', ';\nconsole', ';[/code]']
                    for stop_seq in STOP_SEQS:
                        index = text.find(stop_seq)
                        if index != -1:
                            text = text[:index]
                
                samples.append(dict(task_id=task_id, generation='\n'+text))
            if not os.path.exists('./humaneva-x/results/'+method+'/'+dataset):
                os.makedirs('./humaneva-x/results/'+method+'/'+dataset)
            write_jsonl('./humaneva-x/results/'+method+'/'+dataset+'/'+model+'.jsonl', samples)

if __name__ == '__main__':
    # models = ['ds-1.3', 'ds-6.7', 'codeqwen', 'codegeex4', 'llama3.1']
    model = 'llama3.1'
    main(model)

