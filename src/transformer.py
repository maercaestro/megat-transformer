import torch
import torch.nn as nn
import math
from .attention import MultiHeadAttention,ScaledDotProductAttention
from .utils import FeedForwardNetwork,InputEmbedding,PositionalEncoding
from .encoder import Encoder
from .decoder import Decoder

class Transformer(nn.Module):
    """
    Transformer class: Represents the complete Transformer model with encoder and decoder.
    
    Args:
        num_layers (int): Number of encoder and decoder layers.
        d_model (int): Dimension of the model.
        h (int): Number of attention heads.
        d_ff (int): Dimension of the feed-forward layer.
        src_vocab_size (int): Size of the source vocabulary.
        tgt_vocab_size (int): Size of the target vocabulary.
        max_len (int): Maximum length of the input sequence.
        dropout (float): Dropout probability.
    
    Returns:
        Tensor: Output tensor after the Transformer, shape (batch_size, sequence_length, d_model).
    """
    def __init__(self, num_layers, d_model, h, d_ff, src_vocab_size, tgt_vocab_size, max_len, dropout=0.1):
        super(Transformer, self).__init__()
        self.encoder = Encoder(num_layers, d_model, h, d_ff, src_vocab_size, max_len, dropout)
        self.decoder = Decoder(num_layers, d_model, h, d_ff, tgt_vocab_size, max_len, dropout)
        self.output_layer = nn.Linear(d_model, tgt_vocab_size)

    def forward(self, src, tgt, src_mask=None, tgt_mask=None, memory_mask=None):
        enc_output = self.encoder(src, src_mask)
        dec_output = self.decoder(tgt, enc_output, tgt_mask, memory_mask)
        return self.output_layer(dec_output)