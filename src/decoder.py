import torch
import torch.nn as nn
import math
from .attention import MultiHeadAttention,ScaledDotProductAttention
from .utils import FeedForwardNetwork,InputEmbedding,PositionalEncoding

#build the individual decoder layer first
class DecoderLayer(nn.Module):
    """
    This layer represnts single layer of the transformer decoder

    Argument taken: 
    d_model (int) : Dimension of the model
    h (int) : Number of attention heads
    d_ff (int) : Dimension of the feed-forward layer
    dropout (float) : dropout probability

    returns:
    tensor : output tensor after decoder layer with shape (batch_size, sequence_length, d_model).
    """
    def __init__(self, d_model, h, d_ff, dropout = 0.1):
        super(DecoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(h, d_model)
        self.enc_dec_attn = MultiHeadAttention(h, d_model)
        self.ffn = FeedForwardNetwork(d_model, d_ff)
        self.norm_layers = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(3)])
        self.dropout = nn.ModuleList([nn.Dropout(dropout) for _ in range(3)])


    def forward(self,x, enc_output, tgt_mask = None, memory_mask = None):
        #masked self attention
        x = self.norm_layers[0](x + self.dropout[0](self.self_attn(x,x,x,tgt_mask)))

        #encoder-decoder attention layer
        x = self.norm_layers[1](x+ self.dropout[1](self.enc_dec_attn(x, enc_output,enc_output,memory_mask)))

        #feedforward layer
        x = self.norm_layers[2](x + self.dropout[2](self.ffn(x)))


        return x

#now let's build the decoder block
class Decoder(nn.Module):
    """
    This is the entire decoder block

    argument taken:
    num_layers (int) : Number of decoder layers
    d_model (int) : dimension of the model
    h(int) : number of attention heads
    d_ff (int) : dimension of the feed=forward layer
    vocab_size (int) : size of the vocabulary
    max_len (int) : maximum length of the input sequence
    dropout (float) : dropout probability

    returns:
    tensor: output tensor after the decoder with shape (batch_size, sequence_length, d_model)
    """

    def __init__(self, num_layers, d_model, h, d_ff, vocab_size, max_len, dropout = 0.1):
        super(Decoder, self).__init__()
        self.input_embedding = InputEmbedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_len)
        self.dropout = nn.Dropout(dropout)
        self.layers = nn.ModuleList([DecoderLayer(d_model, h, d_ff, dropout) for _ in range(num_layers)])
        self.norm = nn.LayerNorm(d_model)


    def forward (self, x, enc_output, tgt_mask = None , memory_mask = None):
        x = self.input_embedding(x)
        x = self.pos_encoding(x)
        x = self.dropout(x)
        for layer in self.layers:
            x = layer(x,enc_output,tgt_mask, memory_mask)

        return self.norm(x)