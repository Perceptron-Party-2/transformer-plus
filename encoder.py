#!/usr/bin/env python
# coding: utf-8

# In[1]:


emb_dim = 40
max_seq_len = 1024 #irrelevant
num_heads = 5
drop = 0.1
vocab = 16000
num_blocks = 4


# In[2]:


import torch
import math

pos = torch.arange(0, 5, dtype=torch.float).unsqueeze(1)
div_term = torch.exp(torch.arange(0, 4, 2).float() * -(math.log(10000.0)/4))

result = pos * div_term
print(result.shape)


# In[4]:


class PositionalEncoding(torch.nn.Module):
    def __init__(self, max_seq_len, emb_dim):
        super(PositionalEncoding, self).__init__()
        
        pe = torch.zeros(max_seq_len, emb_dim)
        pos = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1) #unsqueeze makes it [s, 1] instead of [s]
        div_term = torch.exp(torch.arange(0, emb_dim, 2).float() * -(math.log(10000.0)/emb_dim))
        
        pe[:, 0::2] = torch.sin(pos*div_term)
        pe[:, 1::2] = torch.cos(pos*div_term)
        
        self.register_buffer('pe', pe.unsqueeze(0))
        
    def forward(self, x):
        return self.pe[:, :x.size(1)] #[B, x's S, E]


# In[6]:


class MultiHeadAttention(torch.nn.Module):
    def __init__(self, emb_dim, num_heads):
        super(MultiHeadAttention, self).__init__()
        
        self.qkv_proj = torch.nn.Linear(emb_dim, emb_dim * 3)
        self.wo_proj = torch.nn.Linear(emb_dim, emb_dim)
        
        
    def forward(self, x):
        B, S, E = x.shape
        EMBD_HEAD = int(emb_dim / num_heads)

        qry, key, val = self.qkv_proj(x).split(emb_dim, dim=-1) #at this point shapes (B, S, E) = 
        #(B,S,E,embedhead*num_heads)
        #e.g. if E is 30 and num_heads = 5 then emb_head = 6
        qry = qry.reshape(B, S, num_heads, EMBD_HEAD).transpose(1, 2) #split into (B, S, num_heads, emb_head)
        #after transpose, B batches, num_heads per batch, S x emb_head matrices
        key = key.reshape(B, S, num_heads, EMBD_HEAD).transpose(1, 2)
        val = val.reshape(B, S, num_heads, EMBD_HEAD).transpose(1, 2)

       
        att = qry @ key.transpose(-1, -2) / torch.sqrt(torch.tensor(EMBD_HEAD))
        # qry = S x embhead, key = embhead x S and scale with the size of each head
        
        att = torch.nn.functional.softmax(att, dim=-1) #softmax along rows...(B, num_heads, S, S)
        out = (att @ val).transpose(1, 2).reshape(B, S, E) #(B, numheads, S, embhead) to #(B, S, numheads, embhead)
        #to (B,S,E) so per batch, S numhead x embhead matrices, one per sequence word to one S x E matrix
        return self.wo_proj(out)


# In[7]:


class FeedForward(torch.nn.Module):
    def __init__(self, emb_dim, drop):
        super().__init__()
        self.c_fc   = torch.nn.Linear(emb_dim, emb_dim * 4) #By using a higher-dimensional space (e.g., *4),
        #the model can potentially learn more intricate and non-linear relationships within the data
        self.relu   = torch.nn.ReLU()
        self.c_proj = torch.nn.Linear(emb_dim * 4, emb_dim)
        self.drop   = torch.nn.Dropout(drop)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.relu(x)
        x = self.c_proj(x)
        x = self.drop(x)
        return x


# In[8]:


class AttentionBlock(torch.nn.Module):
    def __init__(self, emb_dim, num_heads):
        super().__init__()
        self.ln_1 = torch.nn.LayerNorm(emb_dim)
        self.attn = MultiHeadAttention(emb_dim, num_heads)
        self.ln_2 = torch.nn.LayerNorm(emb_dim)
        self.ffww = FeedForward(emb_dim, drop)

    def forward(self, x):
        x = self.ln_1(x + self.attn(x))
        x = self.ln_2(x + self.ffww(x))
        return x


# In[10]:


class Encoder(torch.nn.Module):
    
    def __init__(self, vocab, emb_dim, drop, num_blocks, num_heads, max_seq_len):
        super().__init__()
        self.tok_emb = torch.nn.Embedding(vocab, emb_dim)
        self.drop    = torch.nn.Dropout(drop)
        self.blocks  = torch.nn.ModuleList([AttentionBlock(emb_dim, num_heads) for _ in range(num_blocks)])
        self.pos = PositionalEncoding(max_seq_len, emb_dim)
        self.vocab.weights = self.tok_emb.weights 

    def forward(self, x):
        
        this_seq_len = min(x.size(1), max_seq_len)
        # Slice the input tensor to the desired sequence length
        x = x[:, :this_seq_len]
        tok_emb = self.tok_emb(x)
        pos_emb = self.pos(x)
        x = tok_emb + pos_emb
        x = self.drop(x)
        for block in self.blocks: x = block(x)
        return x #output (B, S, E)

    def num_params(self):
        gpt_params = sum(p.numel() for p in self.parameters()) #no. of parameters
        emb_params = self.tok_emb.weight.numel() #no of parameters (weights) in the token embeddings
        print(f"Total Parameters: {gpt_params} | Embedding: {emb_params}")
        return { 'gpt_params': gpt_params, 'emb_params': emb_params }
    


# In[ ]:




