'''
Author: Zhengxiang (Jack) Wang 
GitHub: https://github.com/jaaack-wang 
About: Simple BoW model for text matching using pytorch. 
'''
import torch
import torch.nn as nn


class BoW(nn.Module):
    
    def __init__(self, 
                 vocab_size, 
                 output_dim, 
                 embedding_dim=100, 
                 padding_idx=0, 
                 hidden_dim=50, 
                 activation=nn.ReLU()):
        super().__init__()
        
        # summing up the embedding text_embds; mode can also be "mean"
        self.embedding = nn.EmbeddingBag(
            vocab_size, embedding_dim, mode='sum', padding_idx=padding_idx)
        
        self.dense = nn.Linear(embedding_dim * 2, hidden_dim)
        self.activation = activation
        self.dense_out = nn.Linear(hidden_dim, output_dim)
        
    def encoder(self, embd):
        # just to keep the basic structure same throughout
        # but this is basically meaningless here 
        pass
    
    def forward(self, text_a_ids, text_b_ids):
        # shape: text_ids, (batch_size, text_seq_len) 
        # --> text_ids_embd, (batch_size, embedding_dim) 
        # note: this "self.embedding" = "EmbeddingBag"
        text_a_embd = self.embedding(text_a_ids)
        text_b_embd = self.embedding(text_b_ids)

        # concatenate [text_a_embd, text_b_embd]
        # shape: concat, (batch_size, embedding_dim * 2)
        concat = torch.cat([text_a_embd, text_b_embd], dim=-1)

        # go through a dense layer before output
        # shape: hidden_out, (batch_size, hidden_dim)
        hidden_out = self.activation(self.dense(concat))

        # shape: out_logits, (batch_size, output_dim)
        # note that, since we will use cross entropy as the loss func, 
        # we will later use softmax to compute the loss
        out_logits = self.dense_out(hidden_out)
        return out_logits
