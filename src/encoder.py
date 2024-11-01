import torch
import torch.nn as nn
import math
from .attention import MultiHeadAttention,ScaledDotProductAttention
from .utils import FeedForwardNetwork,InputEmbedding,PositionalEncoding

#the encoder layer, combining all parts that we have build previously
class EncoderLayer(nn.Module):
    """
    This class represents a single layer of the transformer encoder

    Arguments:
    d_model (int) : Dimension of the model
    h (int) : number of attention heads
    d_ff (int) : Dimension of the feed forward layer
    dropout (float) :dropout probability of 0.1

    returns:
    tensor : output tensor after the encoder layer, shape (batch_size, sequence_length, d_model).

    """

    def __init__(self, d_model, h, d_ff, dropout = 0.1):
        super(EncoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(h, d_model) #the attention part
        self.ffn = FeedForwardNetwork(d_model, d_ff) #the feedforward network
        self.norm_layers =nn.ModuleList([nn.LayerNorm(d_model) for _ in range(2)]) #don't forget to normnalize after attention and FFN
        self.dropout = nn.ModuleList([nn.Dropout(dropout) for _ in range(2)]) #add dropout after each sublayer



    def forward(self, x, src_mask=None):
        # Self-Attention Layer
        x = self.norm_layers[0](x + self.dropout[0](self.self_attn(x, x, x, src_mask)))
        
        # Feed Forward Layer
        x = self.norm_layers[1](x + self.dropout[1](self.ffn(x)))
        
        return x
    
#and lastly we combine all this layer, including the token embedding and positional embedding to form the entire encoder block
class Encoder(nn.Module):
    """
    The entire encoder block class combining multiple encoder layers and the token and positional embeddings

    argument taken :
    num_layers (int): number of encoder layers
    d_model (int) : dimension of the model
    h (int) : number of attention heads
    d_ff (int): dimension of the feed roward layer
    vocab_size (int) : maximum length of the input sequence
    dropout (float) : dropout probability

    returns:
    tensor: output tensor after the encoder with shape (batch_size, sequence_length,d_model).
    """

    def __init__(self, num_layers, d_model, h, d_ff, vocab_size, max_len, dropout=0.1):
        super(Encoder, self).__init__()
        self.input_embedding = InputEmbedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_len)
        self.dropout = nn.Dropout(dropout)
        self.layers = nn.ModuleList([EncoderLayer(d_model, h, d_ff, dropout) for _ in range(num_layers)])
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x, src_mask=None):
        x = self.input_embedding(x)
        x = self.pos_encoding(x)
        x = self.dropout(x)  # Apply dropout to the sum of embeddings and positional encodings
        for layer in self.layers:
            x = layer(x, src_mask)
        return self.norm(x)
        