import warnings
import torch
import torch.nn as nn
import math

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()

        self.network == nn.Sequential(
            nn.Conv2d(),
            nn.ReLu(),
            nn.Flatten(),
            nn.Linear
        )

        def forward(self, x):
            return self.network(x)


class PatchEmbedding(nn.Module):
    def __init__(self, emb_size):

    def forward(self, x: Tensor) -> Tensor:

        return x


class MultiHeadAttention(nn.Module):
    
    def __init__(self, emb_size, num_heads, dropout):
        """
        emb_size: dimensionality of the input
        num_heads: number of attention heads to split input into
        """
        super().__init__() # alt: super(MultiHeadAttention, self).__init__()
        assert emb_size % num_heads == 0, # must be divisible

        self.emb_size = emb_size
        self.num_heads = num_heads 
        self.keys = nn.Linear(emb_size, emb_size)
        self.queries = nn.Linear(emb_size, emb_size)
        self.values = nn.Linear(emb_size, emb_size)
        self.att_drop = nn.Dropout(dropout)
        self.projection = nn.Linear(emb_size, emb_size)

    def forward(self, x: Tensor, mask: Tensor = None) -> Tensor:
        queries = rearrange(self.queries(x), "b n (h d) -> b h n d", h = self.num_heads)
        values = rearrange(self.values(x), "b n (h d) -> b h n d", h = self.num_heads)
        keys = rearrange(self.keys(x), "b n (h d) -> b h n d", h = self.num_heads)

# rewrite following passage and 

    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        # if provided, apply a mask
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, -1e9)
        attn_probs = torch.softmax(attn_scores, dim=-1)
        output = torch.matmul(attn_probs, V) 
        return output
    
    def split_heads(self, x): # reshape input to have num_heads for multi-head attention
        batch_size, seq_length, emb_size = x.size()
        return x.view(batch_size, seq_length, self.num_heads, self.d_k).transpose(1, 2)
    
    def combine_heads(self, x): # back to original shape
        batch_size, _, seq_length, d_k = x.size()
        return x.transpose(1, 2).contiguous().view(batch_size, seq_length, self.emb_size)
    
    def forward(self, Q, K, V, mask=None):
        Q = self.split_heads(self.W_q(Q))
        K = self.split_heads(self.W_k(K))
        V = self.split_heads(self.W_v(V))

        attn_output = self.scaled_dot_product_attention(Q, K, V, mask)
        output = self.W_o(self.combine_heads(attn_output))
        return output

# code source adapted from: https://github.com/eeyhsong/EEG-Transformer/blob/main/Trans.py
class ResidualConn(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        res = x
        x = self.fn(x, **kwargs)
        x += res
        return x

class FeedForwardBlock(nn.Sequential):
    def __init__(self, emb_size, expansion, drop_p):
        """
        emb_size: dim of model's in- & output
        expansion: dim of inner layer in FFN"""
        super().__init__(
            nn.Linear(emb_size, expansion * emb_size),
            nn.SiLU()
            nn.Dropout(drop_p)
            nn.Linear(expansion * emb_size, emb_size)
        )

# https://docs.pytorch.org/docs/2.13/generated/torch.nn.SiLU.html#torch.nn.SiLU
class SiLU(nn.Module):
    def forward(self, input:Tensor) -> Tensor:
        log_sigmoid = 1 / (1 + np.exp(-input))
        return (input * log_sigmoid)
        

class PositionalEncoding(nn.Module):
    def __init__(self, emb_size, max_seq_length):
        super(PositionalEncoding, self).__init__()
        # missing

    def forward(self, x):
        # add positional encoding to input x
        return x + self.pe[:, :x.size(1)] 
        # 'x.size(1)' to match seq_length of x

class EncoderBlock(nn.Sequential):
    def __init__(self, emb_size, num_heads, expansion, dropout):
        super().__init__(
            ResidualConn(nn.Sequential(
                nn.LayerNorm(emb_size),
                MultiHeadAttention(emb_size, num_heads, dropout),
                nn.Dropout(drop_p)
            )),
            ResidualConn(nn.Sequential(
                nn.LayerNorm(emb_size),
                FeedForwardBlock(
                    emb_size, expansion=forward_expansion, dropout=forward_drop_p),
                    nn.Dropout(dropout)
                )
            ))
            
# do we need a decoder layer?
class DecoderLayer(nn.Module):
    def __init__(self, emb_size, num_heads, expansion, dropout):
        super(DecoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(emb_size, num_heads)
        self.cross_attn = MultiHeadAttention(emb_size, num_heads)
        self.feed_forward = FeedForwardBlock(emb_size, expansion)
        self.norm1 = nn.LayerNorm(emb_size)
        self.norm2 = nn.LayerNorm(emb_size)
        self.norm3 = nn.LayerNorm(emb_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, enc_output, src_mask, tgt_mask):
        attn_output = self.self_attn(x, x, x, tgt_mask)
        x = self.norm1(x + self.dropout(attn_output))
        # cross attention attends to encoder output
        attn_output = self.cross_attn(x, enc_output, enc_output, src_mask)
        x = self.norm2(x + self.dropout(attn_output))
        ff_output = self.feed_forward(x)
        x = self.norm3(x + self.dropout(ff_output))
        return x

# code inspo: https://github.com/reshalfahsi/eeg-motor-imagery-classification/blob/master/EEG_Motor_Imagery_Classification_Using_CNN_Transformer_and_MLP.ipynb

class MLPClassifier(nn.Module):
    def __init__(self, eeg_channel, dropout=0.1):
        super().__init__()
        self.mlp(nn.Sequential(
            nn.Linear(eeg_channel * 2, eeg_channel // 2)
            nn.ReLU(True),
            nn.Dropout(dropout),
            nn.Linear(eeg_channel // 2, 1),
        ))