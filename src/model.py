import warnings
import torch
import torch.nn as nn
import math


"""

IMPORTANT TO DOs:

* check CNN & rewrite Transformer
* check for patchemb whether x has three dimensions or four
* ensure for all classes use of required arguments
* ensure right sequence shapes/dims (shape tracking)
* make sure there are no typos (commata, vars have same name)
* test with small dataset
* visualize in computational graphs


PIPELINE:
1. segment data into fixed-size windows/patches via patch embeddings
2. add positional encodings via patch encoder to preserve order
3. get vector embeddings for transformer via convolutional layers


"""

# ======================================================
# Baseline Model for comparisons and testing pipeline
# ======================================================

class BaselineMLP(nn.Module):

    def __init__(self, input_dim, num_classes, hidden_dim=64, dropout_rate=0.5) -> None:
        # EEG datasets small & noisy -> strong dropout suggested
        # 64 channels/electrodes -> 1:1 mapping
        # call constructor of parent class (nn.Module)
        super(BaselineMLP, self).__init__()
        
        self.layer_1 = nn.Linear(input_dim, hidden_dim)
        self.layer_2 = nn.Linear(hidden_dim, hidden_dim // 2)
        # enforce dense representations (hidden_dim//2)
        self.layer_out = nn.Linear(hidden_dim // 2, num_classes)
        self.dropout = nn.Dropout(dropout_rate)
        self.relu = nn.ReLU()
        
        def forward(self, x):
            x = self.relu(self.layer_1(x))
            x = self.dropout(x)
            x = self.relu(self.layer_2(x))
            x = self.dropout(x)
            x = self.layer_out(x)
            return x

        """ ALTERNATIVE CODE:
                self.network = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout_rate),
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU()
                nn.Dropout(dropout_rate),
                nn.Linear(hidden_dim // 2, num_classes)
                )
        
                def forward(self, x):
                    return self.network(x)"""


# ===================
# Patch Embedding
# ===================

"""dividing our large arrays into smaller patches to reduce the
spatial dimensions of the input for the Transformer (requires fixed-size input)"""

# which patch size is the best?
class PatchEmbedding(nn.Module):
    def __init__(self, in_channels=32, patch_size=32, emb_dim=128):
        super(PatchEmbedding, self).__init__()

        self.patch_size = patch_size
        self.emb_dim = emb_dim
        
        # Linear layer to project flattened patch into higher embedding dim
        self.embedding_layer = nn.Linear(in_channels * patch_size, emb_dim)

    def forward(self, x):
        """
        Args: x: Input EEG data with shape (batch_size, num_channels, num_time_points)
        Returns: embeddings: Patch embeddings with shape (batch_size, num_patches, emb_dim)
        """
        B, C, T = x.shape # batch_size, num_channels, num_time_points
        num_patches = T // self.patch_size 

        if T % self.patch_size != 0:
            raise ValueError(f"Total time points {T} must be divisible by patch size {self.patch_size}")

        # Reshape and permute to form patches
        x = x.view(B, C, num_patches, self.patch_size)
        x = x_patches.permute(0, 2, 1, 3).contiguous() # (batch_size, num_patches, num_channels, num_time_points)

        # Flatten channel and time dimensions -> tokens
        x = x_patches.reshape(B, self.num_patches, C * self.patch_size) # (B, num_patches, in_channels * T)

        # Linear projection to embeded patches
        embeddings = self.embedding_layer(x)

        # alternatively delete num_patches as output (or ensure right dimensionality)
        return x, num_patches


# ===================
# Positional encoding
# ===================

class PositionalEncoding(nn.Module):
    def __init__(self, emb_dim, max_patches):
        super(PositionalEncoding, self).__init__()
        self.emb_dim = emb_dim

        # Positional encoding matrix
        pe = torch.zeros(max_patches, emb_dim)

        # Sinusoidal positional encoding
        # Position indices
        pos = torch.arange(0, max_patches, dtype=torch.float).unsqueeze(1) # (max_patches, 1)

        # Division term (tensor of even indices since pose alternate between sin/cos)
        div_term = torch.exp(torch.arange(0, emb_dim, 2).float() * (-torch.log(torch.tensor(10000.0)) / emb_dim))

        pe[:, 0::2] = torch.sin(pos * div_term) # every second column starting from index 0
        pe[:, 1::2] = torch.cos(pos * div_term) # for all odd indices cosinusoidal values

        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :] # (batch_size, seq_len, d_model)
        return x


# ====
# CNN 
# ====

"""transformation of continuous time-series values into discrete
token sequences/localized patches which self-attention mechanisms 
can then work with"""

class CNN(nn.Module):
    def __init__(self, num_channels, num_classes, emb_dim=128):
        super(CNN, self).__init__()

        # choice between Conv1d and Conv2d was hard -> so why 1d?
        # look into kernel size and stride/padding choices!
        self.conv_layers == nn.Sequential(
            nn.Conv1d(num_channels, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2)


            nn.Conv1d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2)
        )

        self.flatten = nn.Flatten()
        self.embedding_layer = nn.Linear(128 * (num_samples // 4), emb_dim)

        def forward(self, x):
            x = self.conv_layers(x)
            x = self.flatten(x)
            x = self.embedding_layer(x)
            return x
        # what is the final dimension of the output vector


# =====================================
# Different blocks of the Transformer
# =====================================

# Attentional block
class MultiHeadAttention(nn.Module):
    def __init__(self, emb_dim, num_heads, dropout_rate):
        """
        emb_dim: dimensionality of the input
        num_heads: number of attention heads to split input into
        """
        super(MultiHeadAttention, self).__init__()
        assert emb_dim % num_heads == 0, # must be divisible

        self.emb_dim = emb_dim
        self.num_heads = num_heads 
        self.keys = nn.Linear(emb_dim, emb_dim)
        self.queries = nn.Linear(emb_dim, emb_dim)
        self.values = nn.Linear(emb_dim, emb_dim)
        self.att_drop = nn.Dropout(dropout_rate)
        self.projection = nn.Linear(emb_dim, emb_dim)

    def forward(self, x: Tensor, mask: Tensor = None) -> Tensor:
        queries = rearrange(self.queries(x), "b n (h d) -> b h n d", h = self.num_heads)
        values = rearrange(self.values(x), "b n (h d) -> b h n d", h = self.num_heads)
        keys = rearrange(self.keys(x), "b n (h d) -> b h n d", h = self.num_heads)

    # the following was just a code copy for inspiration - try to get inspired and adapt to your needs
    # do we even want scaled dot product attention?

    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        # if provided, apply a mask
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, -1e9)
        attn_probs = torch.softmax(attn_scores, dim=-1)
        output = torch.matmul(attn_probs, V) 
        return output
    
    def split_heads(self, x): # reshape input to have num_heads for multi-head attention
        batch_size, seq_length, emb_dim = x.size()
        return x.view(batch_size, seq_length, self.num_heads, self.d_k).transpose(1, 2)
    
    def combine_heads(self, x): # back to original shape
        batch_size, _, seq_length, d_k = x.size()
        return x.transpose(1, 2).contiguous().view(batch_size, seq_length, self.emb_dim)
    
    def forward(self, Q, K, V, mask=None):
        Q = self.split_heads(self.W_q(Q))
        K = self.split_heads(self.W_k(K))
        V = self.split_heads(self.W_v(V))

        attn_output = self.scaled_dot_product_attention(Q, K, V, mask)
        output = self.W_o(self.combine_heads(attn_output))
        return output

# Residual Connections
# code source adapted from: https://github.com/eeyhsong/EEG-Transformer/blob/main/Trans.py
class ResidualConn(nn.Module):
    def __init__(self, fn):
        super(ResidualConn, self).__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        res = x
        x = self.fn(x, **kwargs)
        x += res # write the syntax in a better way here
        return x

# Activation function
# https://docs.pytorch.org/docs/2.13/generated/nn.SiLU.html#nn.SiLU
class SiLU(nn.Module):
    def forward(self, input:Tensor) -> Tensor:
        log_sigmoid = 1 / (1 + np.exp(-input))
        return (input * log_sigmoid)
    
# Feed-forward block
class FeedForwardBlock(nn.Sequential):
    def __init__(self, emb_dim, expansion, dropout_rate):
        """
        emb_dim: dim of model's in- & output
        expansion: dim of inner layer in FFN"""
        super(FeedForwardBlock, self).__init__()
        self.linear = nn.Linear(emb_dim, expansion * emb_dim)
        self.silu = nn.SiLU()
        self.dropout = nn.Dropout(dropout_rate)
        self.out = nn.Linear(expansion * emb_dim, emb_dim)

    def forward(self, x):
        x = self.linear(x)
        x = self.silu(x)
        x = self.dropout(x)
        x = self.out(x)
        return x


# ==========================================
# Transformer for capturing global relations
# ==========================================

# look here for some code inspo: 
# https://github.com/reshalfahsi/eeg-motor-imagery-classification/blob/master/EEG_Motor_Imagery_Classification_Using_CNN_Transformer_and_MLP.ipynb

"""Encoder-only architecture since the goal is just classification
and does not involve any sequence generation"""

"""
PositionalEncoding? needed? we have it before CNN already...
MultiHeadAttention
# add & norm
FeedForwardBlock
"""

class EncoderBlock(nn.Sequential):
    def __init__(self, emb_dim, num_heads, expansion, dropout_rate):
        super(EncoderBlock, self).__init__()

        self.attention = MultiHeadAttention(emb_dim, num_heads)
        self.norm1 = nn.LayerNorm(emb_dim)

    def forward(self, x):
        x = self.attention(x)
        x = self.norm1(x)
        # add more here
        return x
        """
        ResidualConn(nn.Sequential(
                nn.LayerNorm(emb_dim),
                MultiHeadAttention(emb_dim, num_heads, dropout_rate),
                nn.Dropout(dropout_rate)
            )),
            ResidualConn(nn.Sequential(
                nn.LayerNorm(emb_dim),
                FeedForwardBlock(
                    emb_dim, expansion=forward_expansion, dropout_rate=forward_dropout_rate),
                    nn.Dropout(dropout_rate)
                )
            ))
        """


# ======================
# Final MLP Classifier
# ======================

# ensure the code works - rewritten from self.mlp form

class MLPClassifier(nn.Module):
    def __init__(self, eeg_channel, dropout_rate=0.1):
        super(MLPClassifier, self).__init__()

        self.linear_in = nn.Linear(eeg_channel * 2, eeg_channel // 2)
        self.linear_out = nn.Linear(eeg_channel // 2, 1)
        self.relu = nn.ReLU(True)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        x = self.linear_in(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.linear_out(x)
        return x


# ====================================
# # Full & final Model - EEGClassifier
# ====================================

class EEGClassifier(nn.Module):
    def __init__(self, patch_size=32, in_channels=32, emb_dim=128, 
                 num_patches = 10, out_channels_cnn=64,
                 transfomer_layers=2, num_heads=4, ff_dim=256,
                 expansion = ?, dropout_rate=0.1):
    
        super(EEGClassifier, self).__init__()

         """
            Patch Embedding
            Positional Encoding
            CNN
            EncodingBlock
            MLPClassifier
            """
    
        self.patch_embedding = PatchEmbedding(patch_size, in_channels, emb_dim)
        # positional encodings/patch encoder here
        self.cnn = CNN(emb_dim, out_channels_cnn, ?) # needed? kernel_size=3, stride=1, padding=1)
        self.transformer = EncoderBlock(emb_dim, num_head, expansion, dropout_rate)
        self.mlp = MLPClassifier(out_channels_cnn * (num_patches // 2), 1) # assuming pooling reduces patches by half

    def forward(self, x):
        """
        Args: x: Input EEG data (batch_size, num_channels, num_time_points)
        Returns: logits for classification
        """
        patch_embeddings = self.patch_embedding(x) # shape: (batch_size, num_patches, emb_dim)
        # encoded_patches = self.patch_encoder(patch_embeddings)?
        # where is the rearrange fct?
        transformer_input = rearrange(#encoded_patches, 'b p d -> p b d')
        transformer_output = self.transformer(transformer_input)
        transformer_output = rearrange(transformer_output, 'p b d -> b p d')
        cnn_features = self.cnn(transformer_output)
        flattened_features = rearrange(cnn_features, 'b p d -> b (p d)')
        logits = self.mlp(flattened_features) # shape: (batch_size, 1)
        
        return logits.squeeze() # squeeze to remove single dim for binary classification