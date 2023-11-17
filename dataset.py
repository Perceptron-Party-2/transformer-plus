#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
import tokenizer
import datasets


# In[2]:


class Dataset(torch.utils.data.Dataset):
    def __init__(self, dataset):
        self.ds = dataset
        self.tk_q = tokenizer.Tokenizer('questions')
        self.tk_a = tokenizer.Tokenizer('answers')

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        instruction = self.ds[idx]['instruction']
        sql = self.ds[idx]['response']
        e_text = self.tk_q.encode(instruction)
        d_input = [self.tk_a.sp.bos_id()] + self.tk_a.encode(sql)
        d_target = (self.tk_a.encode(sql)) + [self.tk_a.sp.eos_id()]
        return { 'd_input': torch.tensor(d_input), 'd_target': torch.tensor(d_target), 'e_text': torch.tensor(e_text) }

    def collate_fn(self, batch):
        e_text_pad = torch.nn.utils.rnn.pad_sequence([item['e_text'] for item in batch], batch_first=True, padding_value=self.tk_q.sp.pad_id())
        d_input_pad = torch.nn.utils.rnn.pad_sequence([item['d_input'] for item in batch], batch_first=True, padding_value=self.tk_a.sp.pad_id())
        d_target_pad = torch.nn.utils.rnn.pad_sequence([item['d_target'] for item in batch], batch_first=True, padding_value=self.tk_a.sp.pad_id())
        
        return { 'd_input': d_input_pad, 'd_target': d_target_pad, 'e_text': e_text_pad }


# In[20]:


def dataset_generator(dataset):
    ds = datasets.load_dataset(dataset)['train']

    split = ds.train_test_split(test_size=0.2, shuffle=False)
    train_data = split['train']
    combo_data = split['test']
    split2 = combo_data.train_test_split(test_size = 0.5, shuffle = False)
    test_data = split2['train']
    val_data = split2['test']

    
    train_dataset = Dataset(train_data)
    test_dataset = Dataset(test_data)
    val_dataset = Dataset(val_data)
    
    return train_dataset, val_dataset, test_dataset


# In[30]:


if __name__ == '__main__':
    
    ds_train, ds_val, ds_test = dataset_generator("Clinton/Text-to-sql-v1")
    i = torch.randint(0,len(ds_train), (1,)).item()
    j = torch.randint(0,len(ds_val), (1,)).item()
    k = torch.randint(0,len(ds_test), (1,)).item()
    print('len(train):', len(ds_train))
    print('i = ', i)
    print('train[i]:', ds_train[i])
    print('len(val):', len(ds_val))
    print('j = ', j)
    print('val[i]:', ds_val[j])
    print('len(test):', len(ds_test))
    print('k = ', k)
    print('test[i]:', ds_test[k])
    
# %%

