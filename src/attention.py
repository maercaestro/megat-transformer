import torch
import torch.nn as nn
import math

class ScaledDotProductAttention(nn.Module):
    """
    This class computes attention scores for the input and outputs the weighted sum of values tensor

    Argument taken:

    query (tensor) = Query tensor of shape (batch_size , num_heads, sequence_length , d_k).
    key (tensor) = key tensor of shape (batch_size , num_heads, sequence_length , d_k).
    value (tensor) = key tensor of shape (batch_size , num_heads, sequence_length , d_k).
    mask (tensor, optional) = mask tensor to ignore certain positions with shape (batch_size, 1 ,sequence_length, sequence_length).

    returns:
    Tensor : output tensor after appying attention, shape ((batch_size , num_heads, sequence_length , d_k)
    """

    def __init__(self):
        super(ScaledDotProductAttention,self).__init__()


    def forward(self, query, key , value, mask = None):
        d_k = query.size(-1)
        scores = torch.matmul(query, key.transpose(-2,-1)) / math.sqrt(d_k) #this is the formula above
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9) #in case we want to apply masked. No need for encoder block
        attention = torch.softmax(scores, dim = -1) #apply softmax
        output = torch.matmul(attention,value)
        return output,attention


#now let's use that class for our multi head attention
class MultiHeadAttention(nn.Module):
    """
    This class implements multi-head attention by combining multiple scaled dot product attention modules

    Argument taken:
    h(int) : number of attention heads
    d_model (int) : dimension of the model

    returns:
    tensor: output tensor after concatenating attention frim all heads, shape (batch_size, sequence_length, d_model).

    """
    def __init__(self,h,d_model):
        super(MultiHeadAttention,self).__init__()
        assert d_model % h == 0 # to ensure all head attention to all dimensions
        self.d_k = d_model // h
        self.h = h
        self.linear_layers = nn.ModuleList([nn.Linear(d_model,d_model) for _ in range(4)]) #create 4 linear layers, like the picture below
        self.attention = ScaledDotProductAttention() 
        self.dropout = nn.Dropout(p = 0.1) #apply dropout to the linear layer if required to prevent overfitting


    def forward (self,query,key,value, mask=None):
        batch_size = query.size(0)
        query, key, value = [l(x).view(batch_size, -1, self.h, self.d_k).transpose(1,2) #apply linear transformation for all three k,q,v linear layers
                            for l,x in zip(self.linear_layers, (query,key,value))]
        x, attn = self.attention(query, key, value, mask = mask)
        x= x.transpose(1,2).contiguous().view(batch_size, -1 ,self.h * self.d_k) #prepare for concatenation
        return self.linear_layers[-1](x) #return everything to original d_model dimensions