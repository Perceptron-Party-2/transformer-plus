#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
import tokenizer
import datasets


# In[2]:


class Dataset(torch.utils.data.Dataset):
    def __init__(self):
        self.ds = datasets.load_dataset("Clinton/Text-to-sql-v1")
        self.tk_q = tokenizer.Tokenizer('questions')
        self.tk_a = tokenizer.Tokenizer('answers')

    def __len__(self):
        return len(self.ds['train'])

    def __getitem__(self, idx):
        instruction = self.ds['train'][idx]['instruction']
        sql = self.ds['train'][idx]['response']
        e_text = self.tk_q.encode(instruction)
        d_input = [self.tk_a.sp.bos_id()] + self.tk_a.encode(sql)
        d_target = (self.tk_a.encode(sql)) + [self.tk_a.sp.eos_id()]
        return { 'd_input': torch.tensor(d_input), 'd_target': torch.tensor(d_target), 'e_text': torch.tensor(e_text) }

    def collate_fn(self, batch):
        e_text_pad = torch.nn.utils.rnn.pad_sequence([item['e_text'] for item in batch], batch_first=True, padding_value=self.tk_q.sp.pad_id())
        d_input_pad = torch.nn.utils.rnn.pad_sequence([item['d_input'] for item in batch], batch_first=True, padding_value=self.tk_a.sp.pad_id())
        d_target_pad = torch.nn.utils.rnn.pad_sequence([item['d_target'] for item in batch], batch_first=True, padding_value=self.tk_a.sp.pad_id())
        
        return { 'd_input': d_input_pad, 'd_target': d_target_pad, 'e_text': e_text_pad }


# In[3]:


if __name__ == '__main__':
    i = torch.randint(0,262208, (1,)).item()
    
    ds = Dataset()
    print('len(ds):', len(ds))
    print('i = ', i)
    print('ds[i]:', ds[i])


# In[ ]:




