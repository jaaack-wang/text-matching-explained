'''
Author: Zhengxiang (Jack) Wang 
GitHub: https://github.com/jaaack-wang 
About: Simple CNN model for text matching using pytorch. 
'''
import torch
import torch.nn as nn
import torch.nn.functional as F


class CNN(nn.Module):
    
    def __init__(self, 
                 vocab_size,
                 output_dim,
                 embedding_dim=100,
                 padding_idx=0,
                 num_filter=256,
                 filter_sizes=(3,),
                 hidden_dim=50,
                 activation=nn.ReLU()):

        super().__init__()
        
        self.embedding = nn.Embedding(
            vocab_size, embedding_dim, padding_idx=padding_idx)
        
        self.convs = nn.ModuleList(
            [nn.Conv1d(in_channels=embedding_dim, 
                      out_channels=num_filter, 
                      kernel_size=fz
                     ) for fz in filter_sizes])
        
        self.dense = nn.Linear(len(filter_sizes) * num_filter * 2, hidden_dim)
        self.activation = activation
        self.dense_out = nn.Linear(hidden_dim, output_dim)
        
    def encoder(self, embd):
        # shape: embd, (batch_size, embedding_dim, text_seq_len)
        embd = embd.permute(0, 2, 1)
        # shape: conved (each), (batch_size, filter_size, embedding_dim, kernel_size)
        conved = [self.activation(conv(embd)) for conv in self.convs]
        max_pooled = [F.adaptive_max_pool1d(conv, output_size=1).squeeze(2) for conv in conved]
        # shape: pooled_concat, (batch_size, num_filter * num_filter_sizes)
        pooled_concat = torch.concat(max_pooled, dim=1)
        return pooled_concat
    
    def forward(self, text_a_ids, text_b_ids):
        # shape: text_ids, (batch_size, text_seq_len) 
        # --> text_ids_embd, (batch_size, text_seq_len, embedding_dim) 
        text_a_ids_embd = self.embedding(text_a_ids)
        text_b_ids_embd = self.embedding(text_b_ids)

        # shape: (batch_size, num_filter * num_filter_sizes)
        encoded_a = self.encoder(text_a_ids_embd)
        encoded_b = self.encoder(text_b_ids_embd)

        # concatenate [text_a_embd, text_b_embd]
        # shape: (batch_size, num_filter * num_filter_sizes * 2)
        concat = torch.concat([encoded_a, encoded_b], dim=-1)

        # go through a dense layer before output
        # shape: hidden_out, (batch_size, hidden_dim)
        hidden_out = self.activation(self.dense(concat))

        # shape: out_logits, (batch_size, output_dim)
        # note that, since we will use cross entropy as the loss func, 
        # we will later use softmax to compute the loss
        out_logits = self.dense_out(hidden_out)
        return out_logits
