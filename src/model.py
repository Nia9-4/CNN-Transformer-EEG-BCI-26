import warnings
import torch
import torch.nn as nn
import math


"""

IMPORTANT TO DOs:

* add patch encoder that adds positional encoding or confirm with other code
* ensure all emb_size are embed_dim or vice versa
* ensure for all classes used the required arguments are used
* visualize in computational graphs
* make sure there are no typos
* ensure right sequence shapes (shape tracking)
* remove unnecessary code blocks

"""

# ======================================================
# Baseline Model for comparisons and testing pipeline
# ======================================================

class BaselineMLP(nn.Module):
    """Initialize MLP"""
    def __init__(self, input_size, output_size=1) -> None:
        self.layer_1 = torch.nn.Linear(input_size, 2 * input_size)
        self.layer_2 = torch.nn.Linear(input_size * 2, input_size * 2)
        self.layer_3 = torch.nn.Linear(input_size * 2, input_size)
        self.layer_4 = torch.nn.Linear(input_size, int(input_size) / 4)
        self.layer_out = torch.nn.Linear(int(input_size)/4, output_size)
        self.dropout = torch.nn.Dropout(0.3)
        self.relu = torch.nn.Sigmoid()
    
    def forward(self, x):
        x = self.relu(self.layer_1(x))
        x = self.dropout(x)
        x = self.relu(self.layer_2(x))
        x = self.dropout(x)
        x = self.relu(self.layer_3(x))
        x = self.dropout(x)
        x = self.relu(self.layer_4(x))
        x = self.dropout(x)
        x = self.layer_out(x)
        return x
    
    def train():
        # backward pass included

    def validate():


# ======================================
# CNN - The tokenizer / patch extractor
# ======================================

"""transformation of continuous time-series values into discrete
token sequences/localized patches which self-attention mechanisms 
can then work with"""

class CNN(nn.Module):
    def __init__(self, num_channels, num_classes, embedding_dim=128):
        super(CNN, self).__init__()

        # Convolutional layers
        # choice between Conv1d and Conv2d was hard
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
        self.embedding_layer = nn.Linear(128 * (num_samples // 4), embedding_dim)

        def forward(self, x):
            x = self.conv_layers(x)
            x = self.flatten(x)
            x = self.embedding_layer(x)
            return x


# ===================
# Patch Embedding
# ===================

"""dividing our large arrays into smaller patches to reduce the
spatial dimensions of the input for the Transformer (requires fixed-size input)"""

# unsure whether necessary or adding unncessary complexity
# which patch size is the best?
class PatchEmbedding(nn.Module):
    def __init__(self, patch_size=32, in_channels=32, emb_size=128):
        super(PatchEmbedding, self).__init__()

        # Number of patches
        self.num_patches = (in_channels // patch_size)

        # Ebedding layer to project flattened patch into higher dim
        self.embedding_layer = nn.Linear(patch_size, embed_dim)

    def forward(self, x):
        """
        Args: x: Input EEG data with shape (batch_size, num_channels, num_time_points)
        Returns: embeddings: Patch embeddings with shape (batch_size, num_patches, embed_dims)
        """
        B, C, T = x.shape # batch_size, num_channels, num_time_points

        if C % self.patch_size != 0:
            raise ValueError(f"Number of channels {C} must be divisible by patch size {self.patch_size}")

        # Reshape and permute to form patches
        x_patches = x.view(B, C // self.patch_sizes, self.patch_size, T)
        x_patches = x_patches.permute(0, 2, 1, 3).contiguous() # (B, patch_size, num_patches, T)

        # Flatten channel and time dimensions
        x_patches = x_patches.view(B, self.num_patches, -1) # (B, num_patches, patch_size * T)

        # Linear projection to embeded patches
        embeddings = self.embedding_layer(x_patches)

        return embeddings


# =====================================
# Different blocks of the Transformer
# =====================================

# Attentional block
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

# Residual Connections
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

# Activation function
# https://docs.pytorch.org/docs/2.13/generated/torch.nn.SiLU.html#torch.nn.SiLU
class SiLU(nn.Module):
    def forward(self, input:Tensor) -> Tensor:
        log_sigmoid = 1 / (1 + np.exp(-input))
        return (input * log_sigmoid)
    
# Feed-forward blocks
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
        
# Positional encoding
class PositionalEncoding(nn.Module):
    def __init__(self, emb_size, max_seq_length):
        super(PositionalEncoding, self).__init__()
        # missing

    def forward(self, x):
        # add positional encoding to input x
        return x + self.pe[:, :x.size(1)] 
        # 'x.size(1)' to match seq_length of x


# ==========================================
# Transformer for capturing global relations
# ==========================================

# look here for some code inspo: 
# https://github.com/reshalfahsi/eeg-motor-imagery-classification/blob/master/EEG_Motor_Imagery_Classification_Using_CNN_Transformer_and_MLP.ipynb

"""Encoder-only architecture since the goal is just classification
and does not involve any sequence generation"""

"""
PositionalEncoding
MultiHeadAttention
# add & norm
FeedForwardBlock
"""

class EncoderBlock(nn.Sequential):
    def __init__(self, embed_dim, num_heads, expansion, dropout):
        super(EncoderBlock, self).__init__()

        self.attention = MultiHeadAttention(embed_dim, num_heads)
        self.norm1 = nn.LayerNorm(embed_dim)
        
        """
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
        """


# ======================
# Final MLP Classifier
# ======================

class MLPClassifier(nn.Module):
    def __init__(self, eeg_channel, dropout=0.1):
        super().__init__()

        self.mlp(nn.Sequential(
            nn.Linear(eeg_channel * 2, eeg_channel // 2)
            nn.ReLU(True),
            nn.Dropout(dropout),
            nn.Linear(eeg_channel // 2, 1),
        ))


# ====================================
# # Full & final Model - EEGClassifier
# ====================================

class EEGClassifier(nn.Module):
    def __init__(self, patch_size=32, in_channels=32, embed_dim=128, 
                 num_patches = 10, out_channels_cnn=64,
                 transfomer_layers=2, num_heads=4, ff_dim=256,
                 expansion = ?, dropout_rate=0.1):
        super(EEGClassifier, self).__init__()

         """
            CNN
            EncodingBlock
            MLPClassifier
            """
    
        self.patch_embedding = PatchEmbedding(patch_size, in_channels, embed_dim)
        # Patch Encoder here?
        self.cnn = CNN(embed_dim, out_channels_cnn, # needed? kernel_size=3, stride=1, padding=1)
        self.transformer = EncoderBlock(embed_dim, num_head, expansion, dropout_rate)
        self.mlp = MLPClassifier(out_channels_cnn * (num_patches // 2), 1) # assuming pooling reduces patches by half

    def forward(self, x):
        """
        Args: x: Input EEG data (batch_size, num_channels, num_time_points)
        Returns: logits: ...for classification
        """
        patch_embeddings = self.patch_embedding(x) # shape: (batch_size, num_patches, embed_dim)
        # encoded_patches = self.patch_encoder(patch_embeddings)?
        transformer_input = rearrange(#encoded_patches, 'b p d -> p b d')
        transformer_output = self.transformer(transformer_input)
        transformer_output = rearrange(transformer_output, 'p b d -> b p d')
        cnn_features = self.cnn(transformer_output)
        flattened_features = rearrange(cnn_features, 'b p d -> b (p d)')
        logits = self.mlp(flattened_features) # shape: (batch_size, 1)
        
        return logits.squeeze() # squeeze to remove single dim for binary classification