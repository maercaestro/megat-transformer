import torch
import torch.nn as nn
import math


# Input Embedding layer for encoder
class InputEmbedding(nn.Module):
    """
    InputEmbedding class: Maps token indices to dense vectors of size (batch_size, sequence_length, d_model).
    
    Args:
        vocab_size (int): Size of the vocabulary.
        d_model (int): Dimension of the embedding vectors.
    
    Returns:
        Tensor: Embedded representation of input tokens with shape (batch_size, sequence_length, d_model).
     """
    def __init__(self,vocab_size,d_model):
        super(InputEmbedding,self).__init__()
        self.embedding = nn.Embedding (vocab_size,d_model) #we just use the embedding function from pytorch

    def forward(self,x):
        return self.embedding(x)

#positional encoding layer for encoder block
class PositionalEncoding(nn.Module):
    """
    Positional encoder class : add position information to our data coming from token embeddings to retain the sequence order

    Argument taken:
    d_model (int) : Dimensions of the embedding vectors
    max_len (int) : Maximum length of the input sequence


    Returns:
    Tensor : Input tensor with posisitonal encodings added, with shape (sequence_length, batch_size, d_model)
    This tensor won't be updated during training, so we will register is as buffer
    """
    def __init__(self,d_model,max_len = 5000):
        super(PositionalEncoding, self).__init__()
        #initialize the pe tensor first
        pe= torch.zeros (max_len, d_model)
        position = torch.arange(0, max_len, dtype= torch.float).unsqueeze(1) #the position of our embedding token
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0)/d_model)) #the division
        pe[: , 0 :: 2] = torch.sin(position * div_term) #when the dimension number is even
        pe[:,  1 :: 2] = torch.cos(position * div_term) #when the dimension number is odd
        pe = pe.unsqueeze(0).transpose(0,1)
        self.register_buffer('pe', pe) #to ensure this number won't be updated during training

    def forward(self,x):
        x = x + self.pe[:x.size(0),:]
        return x
    

#build the feedforward network in our encoder block
class FeedForwardNetwork(nn.Module):
    """
    This class applies two linear transformations with a ReLU activation function in between

    Argument taken:
    d_model (int) : Dimension of the model
    d_ff (int): Dimension of the feed-forward layer

    returns: 
    Tensor: Output tensor after the feed forward network, shape (batch_size, sequence_length, d_model).

    """

    def __init__(self, d_model , d_ff = 2048):
        super(FeedForwardNetwork , self).__init__()
        self.linear1 = nn.Linear(d_model,d_ff) #the first layer, changing the dimensions from 512 to 2048
        self.linear2 = nn.Linear(d_ff, d_model) # the second layer, changing back the dimensions to 512
        self.relu = nn.ReLU() #the ReLU activation functions
        self.dropout = nn.Dropout(p=0.1) 

    def forward(self,x):
        return self.linear2(self.dropout(self.relu(self.linear1(x))))