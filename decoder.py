#!/usr/bin/env python
# coding: utf-8

# # Transformer Model

# Import libraries

# In[47]:


import torch
import torch.nn as nn


# The embedding layer

# In[65]:


class EmbeddingLayer(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super(EmbeddingLayer, self).__init__()
        self.embed = nn.Embedding(vocab_size, embedding_dim)
        self.embedding_dim = embedding_dim
        
    def positional_embedding(self, text):
        batch_size = text.size(0)
        sequence_length = text.size(1) 
        embedding_dim = self.embedding_dim
        position = torch.arange(0, sequence_length).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, embedding_dim, 2).float() * -(torch.log(torch.tensor(10000.0)) / embedding_dim))
        encoding = torch.zeros(sequence_length, embedding_dim)
        encoding[:, 0::2] = torch.sin(position * div_term)
        encoding[:, 1::2] = torch.cos(position * div_term[:embedding_dim // 2])
        return encoding

    def forward(self, text):
        embedded_text = self.embed(text)
        positionally_embedded_text = embedded_text + self.positional_embedding(text)
        return positionally_embedded_text


# In[79]:


# emb = EmbeddingLayer(8, 5)
# print(emb(torch.tensor([[1, 3, 5, 7]])))


# (Masked) Multi-head attention

# In[ ]:


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, forward_masked):
        super(MultiHeadAttention, self).__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.forward_masked = forward_masked

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

    def forward(self, query, key, value):
        batch_size = query.size(0)
        query_length = query.size(1)
        key_length = key.size(1)
        value_length = value.size(1)
        
        # Linear projections
        query = self.W_q(query)
        key = self.W_k(key)
        value = self.W_v(value)

        # Split into num_heads
        query = query.view(batch_size, query_length, self.num_heads, self.head_dim).transpose(1, 2)
        key = key.view(batch_size, key_length, self.num_heads, self.head_dim).transpose(1, 2)
        value = value.view(batch_size, value_length, self.num_heads, self.head_dim).transpose(1, 2)

        # Scaled Dot-Product Attention
        attention_scores = torch.matmul(query, key.transpose(-2, -1)) / torch.sqrt(torch.tensor(self.head_dim, dtype=torch.float32))

        # Apply mask if provided
        if forward_masked == True:
            mask = torch.tril(torch.ones_like(attention_scores))
            attention_scores = attention_scores.masked_fill(mask == 0, float('-inf'))

        attention_weights = torch.nn.functional.softmax(attention_scores, dim=-1)
        attention_output = torch.matmul(attention_weights, value)

        # Merge heads
        attention_output = attention_output.transpose(1, 2).contiguous().view(batch_size, value_length, self.d_model)

        # Linear projection
        output = self.W_o(attention_output)

        return output


# In[76]:


#a = torch.nn.functional.softmax(torch.ones(2,3), dim=-1)


# In[81]:


#torch.matmul(a, torch.ones(3,2))


# The feed forward layer

# In[ ]:


class FeedForward(nn.Module):
    def __init__(self, d_model, ff_dim):
        super(FeedForward, self).__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        return x


# The encoder

# In[ ]:


class Encoder(nn.Module):
    def __init__(self, vocab_size, embedding_dim, ff_dim, num_heads, num_layers, dropout):
        super(Encoder, self).__init__()
        
        self.num_layers = num_layers
        
        self.dropout = nn.Dropout(dropout)
        
        self.embed = EmbeddingLayer(vocab_size, embedding_dim)

        self.mha = MultiHeadAttention(embedding_dim, num_heads, forward_masked = False)

        self.lnorm = nn.LayerNorm(embedding_dim)
        
        self.ff = FeedForward(embedding_dim, ff_dim)
        
    def forward(self, text):
        
        embedded_text = self.embed(text)
        embedded_text = self.dropout(embedded_text)
        
        for i in range(self.num_layers):
            attended_text = self.mha(query = embedded_text, key = embedded_text, value = embedded_text)
            attended_text = self.dropout(attended_text)
            attended_text = self.lnorm(embedded_text + attended_text)
            
            ff_text = self.ff(attended_text)
            ff_text = self.dropout(ff_text)
            ff_text = self.lnorm(attended_text + ff_text)

            embedded_text = ff_text
        
        return embedded_text


# The decoder

# In[63]:


class Decoder(nn.Module):
    def __init__(self, vocab_size, embedding_dim, ff_dim, num_heads, num_layers, dropout, with_encoder):
        super(Decoder, self).__init__()
        
        self.num_layers = num_layers
        
        self.with_encoder = with_encoder
        
        self.embed = EmbeddingLayer(vocab_size, embedding_dim)
        
        self.masked_mha = MultiHeadAttention(embedding_dim, num_heads, forward_masked = True)
        
        self.lnorm = nn.LayerNorm(embedding_dim)
        
        self.dropout = nn.Dropout(dropout)
        
        self.mha = MultiHeadAttention(embedding_dim, num_heads, forward_masked = False)
        
        self.ff = FeedForward(embedding_dim, ff_dim)
        
        self.linear = nn.Linear(embedding_dim, vocab_size)
        self.linear.weights = self.embed.weights
        
        self.softmax = nn.Softmax()
    
    def forward(self, text, encoder_output=None):
        
        embedded_text = self.embed(text)
        embedded_text = self.dropout(embedded_text)
        
        for i in range(self.num_layers):
            attended_text = self.masked_mha(query = embedded_text, key = embedded_text, value = embedded_text)
            attended_text = self.dropout(attended_text)
            attended_text = self.lnorm(embedded_text + attended_text)
            
            if self.with_encoder == True:
                cross_attended_text = self.mha(query = attended_text, key = encoder_output, value = encoder_output)
                cross_attended_text = self.dropout(cross_attended_text)
                cross_attended_text = self.lnorm(cross_attended_text + attended_text)
                attended_text = cross_attended_text
            
            ff_text = self.ff(attended_text)
            ff_text = self.dropout(ff_text)
            ff_text = self.lnorm(attended_text + ff_text)

            embedded_text = ff_text
        
        logits = self.linear(embedded_text)
        
        probabilities = self.softmax(logits)
        
        return probabilities


# The transformer

# In[ ]:


class Transformer(nn.Module):
    def __init__(self, vocab_size, embedding_dim, ff_dim, num_heads, num_layers, dropout, with_encoder):
        super(Transformer, self).__init__()
        self.with_encoder = with_encoder
        if with_encoder == True:
            self.encoder = Encoder(vocab_size, embedding_dim, ff_dim, num_heads, num_layers, dropout)
        self.decoder = Decoder(vocab_size, embedding_dim, ff_dim, num_heads, num_layers, dropout, with_encoder)

    def forward(self, text):
        if self.with_encoder == True:
            encoder_output = self.encoder(text)
            probabilities = self.decoder(text, encoder_output)
        else:    
            probabilities = self.decoder(text, encoder_output = None)
        return x

