'''
Author: Zhengxiang (Jack) Wang 
GitHub: https://github.com/jaaack-wang 
About: Simple CNN model for text matching using tensorflow. 
'''
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow as tf


class CNN(keras.Model):
    
    def __init__(self,
                 vocab_size,
                 output_dim,
                 embedding_dim=100,
                 mask_zero=True,
                 num_filter=256,
                 filter_sizes=(3,),
                 hidden_dim=50,
                 activation=layers.ReLU(), 
                 activation_out=tf.sigmoid):
        
        super().__init__()
        
        self.embedding = layers.Embedding(
            vocab_size, embedding_dim, mask_zero=mask_zero)
        
        self.convs = [layers.Conv1D(
                filters=num_filter, 
                kernel_size=fz
            ) for fz in filter_sizes]
        
        self.num_filter = num_filter
        self.filter_sizes = filter_sizes        
        self.dense = layers.Dense(hidden_dim)
        self.activation = activation
        self.dense_out = layers.Dense(output_dim)
        self.activation_out = activation_out
    
    def encoder(self, embd):        
        # shape: conved (each), (batch_size, embedding_dim-X, num_filter)
        conved = [self.activation(conv(embd)) for conv in self.convs]
        # shape: max_pooled, (batch_size, num_filter)
        max_pooled = [layers.GlobalMaxPool1D()(conv) for conv in conved]
        # shape: pooled_concat, (batch_size, num_filter * num_filter_sizes)
        pooled_concat = tf.concat(max_pooled, axis=1)
        return pooled_concat  
    
    def call(self, inputs):
  
        text_a_ids, text_b_ids = inputs
    
        # shape: text_ids, (batch_size, text_seq_len) 
        # --> text_ids_embd, (batch_size, text_seq_len, embedding_dim) 
        text_a_ids_embd = self.embedding(text_a_ids)
        text_b_ids_embd = self.embedding(text_b_ids)

        # shape: encoded, (batch_size, num_filter * num_filter_sizes)
        encoded_a = self.encoder(text_a_ids_embd)
        encoded_b = self.encoder(text_b_ids_embd)

        # concatenate [encoded_a, encoded_b]
        # shape: concat, (batch_size, embedding_dim * 2)
        concat = tf.concat([encoded_a, encoded_b], axis=-1)

        # go through a dense layer before output
        # shape: hidden_out, (batch_size, hidden_dim)
        hidden_out = self.activation(self.dense(concat))

        # shape: out_logits, (batch_size, output_dim)
        out_logits = self.dense_out(hidden_out)
        
        # for binary, use "tf.sigmoid" as the output activation func; 
        # for multi-class, use "tf.math.softmax" instead, and
        # you also need to one-hot encode your labels as well
        return self.activation_out(out_logits)
