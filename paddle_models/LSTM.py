'''
Author: Zhengxiang (Jack) Wang 
GitHub: https://github.com/jaaack-wang 
About: Simple LSTM model for text matching using paddle. 
'''

import paddle 
import paddle.nn as nn


class LSTM(nn.Layer):

    def __init__(self,
                 vocab_size,
                 output_dim,
                 embedding_dim=100,
                 lstm_hidden_dim=128,
                 padding_idx=0,
                 hidden_dim_out=50,
                 n_layers=1,
                 bidirectional=False,
                 dropout_rate=0.0,
                 activation=nn.ReLU()):
        
        super().__init__()

        self.embedding = nn.Embedding(
            vocab_size, embedding_dim, padding_idx=padding_idx)
        
        self.direction = 'bidirect' if bidirectional is True else 'forward'
        self.lstm = nn.LSTM(
            embedding_dim, lstm_hidden_dim, n_layers, self.direction, dropout=dropout_rate)
        
        lstm_out_dim = lstm_hidden_dim * 2 if bidirectional is True else lstm_hidden_dim
        self.dense = nn.Linear(lstm_out_dim * 2, hidden_dim_out)
        self.activation = activation
        self.dense_out = nn.Linear(hidden_dim_out, output_dim)


    def encoder(self, embd, seq_len):
        # shape: encoded, (batch_size, seq_len, lstm_hidden_dim)
        # shape: hidden, (1, batch size, lstm_hidden_dim)
        # shape: there is a "cell" here!
        encoded, (hidden, cell) = self.lstm(embd, sequence_length=seq_len)
        
        # shape: hidden, (batch size, lstm_out_dim);
        # if bidirectional, lstm_out_dim = lstm_hidden_dim * 2; otherwise same
        if self.direction != 'bidirect':
#             return  hidden[-1]    # This works too
            return hidden[-1, :, :]
#         return paddle.concat([hidden[-2], hidden[-1]], axis=-1)    # This works too
        return paddle.concat((hidden[-2, :, :], hidden[-1, :, :]), axis=1)

    def forward(self, text_a_ids, text_b_ids, text_a_seq_len, text_b_seq_len):
        # shape: text_ids, (batch_size, text_seq_len) 
        # --> text_ids_embd, (batch_size, text_seq_len, embedding_dim)  
        text_a_ids_embd = self.embedding(text_a_ids)
        text_b_ids_embd = self.embedding(text_b_ids)

        # shape: (batch_size, lstm_out_dim)
        encoded_a = self.encoder(text_a_ids_embd, text_a_seq_len)
        encoded_b = self.encoder(text_b_ids_embd, text_b_seq_len)

        # concatenate [text_a_embd, text_b_embd]
        # shape: (batch_size, lstm_out_dim * 2)
        concat = paddle.concat([encoded_a, encoded_b], axis=-1)
        
        # go through a dense layer before output
        # shape: hidden_out, (batch_size, hidden_dim_out)
        hidden_out = self.activation(self.dense(concat))

        # shape: out_logits, (batch_size, output_dim)
        # note that, since we will use cross entropy as the loss func, 
        # we will later use softmax to compute the loss
        out_logits = self.dense_out(hidden_out)
        return out_logits
