#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import torch
import tokenizer
import datasets


class Dataset(torch.utils.data.Dataset):
    def __init__(self):
        self.ds = datasets.load_dataset("Clinton/Text-to-sql-v1")
        self.tk = tokenizer.Tokenizer()
        self.tk.load()

    def __len__(self):
        return len(self.ds['train'])

    def __getitem__(self, idx):
        e_text = self.ds['train'][idx]['instruction']
        sql = self.ds['train'][idx]['response']
        d_input = [self.tk.sp.bos_id()] + self.tk.encode(sql)
        d_target = (self.tk.encode(sql)) + [self.tk.sp.eos_id()]
        return { 'd_input': torch.tensor(d_input), 'd_target': torch.tensor(d_target), 'e_text': torch.tensor(e_text) }

    def collate_fn(self, batch):
        e_text_pad = torch.nn.utils.rnn.pad_sequence([item['e_text'] for item in batch], batch_first=True, padding_value=self.tk.sp.pad_id())
        d_input_pad = torch.nn.utils.rnn.pad_sequence([item['d_input'] for item in batch], batch_first=True, padding_value=self.tk.sp.pad_id())
        d_target_pad = torch.nn.utils.rnn.pad_sequence([item['d_target'] for item in batch], batch_first=True, padding_value=self.tk.sp.pad_id())
        
        return { 'd_input': input_pad, 'd_target': label_pad, 'e_text': e_text_pad }

