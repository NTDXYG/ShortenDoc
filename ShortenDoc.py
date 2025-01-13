from typing import *
import logging
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import re

class ShortenDoc():
    def __init__(
        self, 
        model,
        tokenizer,
        threshold: Optional[float] = 0.999,
        topk: Optional[int] = 20,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.threshold = threshold
        self.tokenizer = tokenizer
        self.model = model
        self.topk = topk
        self.LM = CodeLLM(self.model, self.tokenizer)
        self.reduced = ['the', 'a', 'that', 'The', 'and', 'are', 'an', 'is', 'of', 'in', 
                        'there', 'which', 'to', 'You', 'it', 'you', 'be', 'There', 'any', 'then']
        self.remove_texts = ['`', 'Write a python function to ', 'Write a function to ',
                             'Write a python function ', 'Write a function ']
        
    def shorten(
            self,
            original_text: str,
            template: str
    ):
        torch.cuda.empty_cache()
        raw_text = original_text
        candi_text_list = []
        for text in self.remove_texts:
            original_text = original_text.replace(text, '')
        original_text = re.sub(r"\s+", " ", original_text)
        original_text = original_text.strip().capitalize()
        for reduce_token in tqdm(self.reduced, desc='Removing Stop Words...'):
            original_text_tokens = original_text.split()
            if reduce_token in original_text_tokens:
                original_text_tokens.remove(reduce_token)
                new_text = ' '.join(original_text_tokens)
                if self.cal_similar(self.get_embedding(original_text, template, eos_token=True), 
                                    self.get_embedding(new_text, template, eos_token=True)) >= self.threshold:
                    original_text = new_text
        self.raw_text_embedding = self.get_embedding(original_text, template)
        self.ppl_tokens = []
        self.raw_tokens = []
        self.remove_token_idxs = []
        self.raw_tokens = self.tokenizer.tokenize(original_text)
        split_len = len(self.raw_tokens)
        self.ppl_tokens = self.get_ppl_tokens(self.raw_tokens, template)
        processed_text = original_text
        for i in tqdm(range(split_len), desc='Shortening the DocString'):
            candidates_remove_tokens_index = []
            temp_list = []
            for idx in self.ppl_tokens[:self.topk]:
                temp_list.extend([idx])
                candidates_remove_tokens_index.append(temp_list.copy())
                if(len(temp_list) > 1):
                    candidates_remove_tokens_index.append(temp_list.copy()[1:])
                if(len(temp_list) > 2):
                    candidates_remove_tokens_index.append(temp_list.copy()[2:])
                if(len(temp_list) > 3):
                    candidates_remove_tokens_index.append(temp_list.copy()[3:])

            for index_list in candidates_remove_tokens_index:
                text = self.get_processed_text(self.raw_tokens, index_list).strip()
                text = re.sub(r"\s+", " ", text)
                text_embedding = self.get_embedding(text, template)
                processed_text = text
                if self.cal_similar(text_embedding, self.raw_text_embedding) >= self.threshold:
                    for index in index_list:
                        self.ppl_tokens.remove(index)
                        self.remove_token_idxs.append(index)
                    candi_text_list.append(processed_text)
                    del text_embedding
                    break
            if len(candi_text_list) == 0:
                break
        if len(candi_text_list) == 0:
            return [original_text], 1-len(self.tokenizer.tokenize(original_text))/len(self.tokenizer.tokenize(raw_text))

        candi_text_list.sort(key=lambda x: len(self.tokenizer.tokenize(x)), reverse=True)
        return candi_text_list, 1-len(self.tokenizer.tokenize(candi_text_list[-1]))/len(self.tokenizer.tokenize(raw_text))

    def get_embedding(self, text, template='', eos_token=False):
        if eos_token:
            text_encoding = self.tokenizer(template + text + '\n    \"\"\"' + self.tokenizer.eos_token)['input_ids']
        else:
            text_encoding = self.tokenizer(template + text + '\n    \"\"\"\n')['input_ids']
        if len(text_encoding) > 512:
            encoding = text_encoding[:512]
            if eos_token:
                encoding.extend(self.tokenizer.encode('\n    \"\"\"' + self.tokenizer.eos_token))
            else:
                encoding.extend(self.tokenizer.encode('\n    \"\"\"\n'))
        else:
            encoding = text_encoding
        encoding = torch.LongTensor(encoding).unsqueeze(0).to(self.model.device)
        with torch.no_grad():
            text_output = self.model(encoding, output_hidden_states=True).hidden_states[-1][:, -1, :]
        return text_output
        
    def cal_similar(self, processed_text_output, original_text_output):
        norm_processed = processed_text_output.norm(dim=-1, keepdim=True)
        norm_original = original_text_output.norm(dim=-1, keepdim=True)
        sim = (processed_text_output @ original_text_output.T) / (norm_processed * norm_original)
        return sim.item()

    def get_ppl_tokens(self, orig_text_split, template):
        def filter_sent(split_sent, pos):
            words_list = split_sent[: pos] + split_sent[pos + 1:]
            return self.tokenizer.decode([self.tokenizer.convert_tokens_to_ids(token) for token in words_list])

        def get_PPL(split_text):
            text_length = len(split_text)
            processed_sents = []
            for i in range(text_length):
                processed_sents.append(template + filter_sent(split_text, i) + '\n    \"\"\"\n')
            ppl_li_record = []
            processed_sents = DataLoader(processed_sents, batch_size=1, shuffle=False) # len=len(split_text)+1
            for idx, sents in enumerate(processed_sents):
                ppl_li_record.append(self.LM(sents, self.raw_text_embedding))
            return ppl_li_record
        
        ppl_li_record = get_PPL(orig_text_split)
        sorted_ppl_record = sorted(enumerate(ppl_li_record), key=lambda x: x[1])
        return [t[0] for t in sorted_ppl_record]
        
    def get_processed_text(self, orig_text_split, idx_list):
        temp_text_split = orig_text_split.copy()
        temp_remove_list = []
        temp_remove_list.extend(self.remove_token_idxs)
        for idx in idx_list:
            temp_remove_list.append(idx)
        sorted_remove_list = sorted(temp_remove_list, reverse=True)
        for remove_index in sorted_remove_list:
            del temp_text_split[remove_index]
        orig_text = self.tokenizer.decode([self.tokenizer.convert_tokens_to_ids(token) for token in temp_text_split])
        return orig_text

class CodeLLM():
    def __init__(self, model, tokenizer):
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.tokenizer = tokenizer
        self.lm = model.to(self.device)
        self.tokenizer.pad_token = self.tokenizer.eos_token
    
    def __call__(self, sents, raw_text_embedding):
        logging.getLogger("transformers").setLevel(logging.ERROR)
        encoding = self.tokenizer(sents, return_tensors="pt").to(self.device)
        input_ids = encoding['input_ids']
        with torch.no_grad():
            output = self.lm(input_ids)
            logits = output.logits
            probs = torch.softmax(logits, dim=-1)
            self_info = -torch.log(probs)
            input_ids_expaned = input_ids[:, 1:].unsqueeze(-1)
            self_info = self_info[:, :-1].gather(-1, input_ids_expaned).squeeze(-1).squeeze(0).tolist()
        return np.mean(self_info)

