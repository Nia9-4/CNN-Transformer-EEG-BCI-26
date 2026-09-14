import warnings
import numpy as np
import torch
import torch.nn as nn
import math


class PoorMLP(nn.Module):
    """
    A small and simple baseline model to compare the main model to
    
    Parameters:
    - in_dim: Dimension of the input data
    - num_classes: Number of classes (2)
    - hidden_dim: Hidden dimension size of MLP
    - dropout: Dropout rate, here large since comp. expensive
    """

    def __init__(self, in_dim, num_classes=2, hidden_dim=128, dropout=0.5) -> None:
        # EEG datasets small & noisy -> strong dropout suggested
        # 64 channels/electrodes -> 1:1 mapping
        super(PoorMLP, self).__init__()
        
        self.layer_1 = nn.Linear(in_dim, hidden_dim)
        self.layer_2 = nn.Linear(hidden_dim, hidden_dim // 2)
        # Enforce dense representations (hidden_dim // 2)
        self.layer_out = nn.Linear(hidden_dim // 2, num_classes)
        self.dropout = nn.Dropout(dropout)
        self.elu = nn.ELU()
        
    def forward(self, x):
        x = torch.mean(x, dim=2) # (batch, channels)
        x = self.relu(self.layer_1(x))
        x = self.dropout(x)
        x = self.elu(self.layer_2(x))
        x = self.dropout(x)
        x = self.layer_out(x)
        return x
    

# ======================================================
# Baseline Model for comparisons and testing pipeline
# ======================================================

class BaselineCNN(nn.Module):
    def __init__(self, num_classes=2):
        super(BaselineCNN, self).__init__()

        self.conv1 = nn.Conv1d(in_dim=64, out_dim=16, kernel_size=25)
        self.pool = nn.AdaptiveAvgPool1d(1) # Reduce time dimension to 1
        self.fc = nn.Linear(16, num_classes)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.conv1(x)) # (batch, 16, samples-24)
        x = self.pool(x).squeeze(-1) # (batch, 16)
        x = self.fc(x)               # (batch, 2)
        return x


# ====
# CNN 
# ====

"""transformation of continuous time-series values into discrete
token sequences/localized patches which self-attention mechanisms 
can then work with"""

class CNN(nn.Module):
    """
    CNN to capture local relationships before input is passed to transformer

    Parameters:
    - n_channels: Number of channels, 64 in our dataset
    - num_classes: Number of classes, 2
    - emb_dim: Dimension of embeddings
    - fs: Sampling rate
    """

    def __init__(self, n_channels=64, num_classes=2, emb_dim=128, fs=160):
        super(CNN, self).__init__()

        F1 = 40 # Number of temporal filters
        D = 2 # Depth multiplier for spatial filters
        k_t = fs // 10 # Temporal kernel size, here: 16

        # Temporal Convolution: Learn frequency/band-pass filters
        # input: (batch, 1, n_channels, time)
        self.temp_conv = nn.Conv2d(1, F1, (1, k_t), padding=(0, k_t // 2), bias=False)
        self.bn1 = nn.BatchNorm2d(F1)

        # Spatial Convolution: Learn spatial filters
        self.spat_conv = nn.Conv2d(F1, F1 * D, (n_channels, 1), groups=F1, bias=False)
        self.bn2 = nn.BatchNorm2d(F1 * D)

        self.elu = nn.ELU(True)
        self.pool = nn.AvgPool2d(kernel_size=(1, 4)) # Pool along time dimension
        self.embedding_layer = nn.Linear(F1 * D, emb_dim)

    def forward(self, x):
        # x input shape: (batch, n_channels, time)
        # Reshape to (batch, 1, n_channels, time) for Conv2d
        x = x.unsqueeze(1)

        x = self.temp_conv(x)
        x = self.bn1(x)
        x = self.elu(x)

        x = self.spat_conv(x)
        x = self.bn2(x)
        x = self.elu(x)

        x = self.pool(x) # (batch, F1*D, 1, time_reduced)

        # Transform to 3D for transformer
        x = x.squeeze(2) # (batch, F1*D, time_reduced)
        x = x.transpose(1, 2) # (batch, time_reduced, F1*D)

        x = self.embedding_layer(x) # (batch, time_reduced, emb_dim)

        return x


# ===================
# Positional encoding
# ===================

class PositionalEncoding(nn.Module):
    """
    Positional encoding so important temporal information 
    is not lost when input is passed onto Transformer

    Parameters:
    - emb_dim: Dimension of embeddings
    - max_patches: Maximum number of patches
    """
    
    def __init__(self, emb_dim, max_patches):
        super(PositionalEncoding, self).__init__()
        self.emb_dim = emb_dim
        assert emb_dim % 2 == 0 # must be even for sin/cos pairs

        # Positional encoding matrix
        pe = torch.zeros(max_patches, emb_dim)

        # Sinusoidal positional encoding
        # Position indices
        pos = torch.arange(0, max_patches, dtype=torch.float).unsqueeze(1) # (max_patches, 1)

        # Division term (tensor of even indices since pose alternate between sin/cos)
        div_term = torch.exp(torch.arange(0, emb_dim, 2).float() * (-math.log(10000.0) / emb_dim))

        pe[:, 0::2] = torch.sin(pos * div_term) # Every second column starting from index 0
        pe[:, 1::2] = torch.cos(pos * div_term) # For all odd indices cosinusoidal values

        # New batch dimension: (1, max_patches, emb_dim)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :] # (batch, seq_len, d_model)
        return x


# =====================================
# Different Transformer components
# =====================================

class ResidualConnection(nn.Module):
    """
    Residual connection for faster and more efficient Transformer training
    
    Parameters:
    - block: Network the residual connection is applied to (e. g. feed forward block)
    - emb_dim: Embedding dimension
    - dropout: Dropout rate
    """

    def __init__(self, block, emb_dim, dropout=0.1):
        super(ResidualConnection, self).__init__()
        self.block = block
        self.norm = nn.LayerNorm(emb_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.norm(x + self.block(x))
        x = self.dropout(x)
        return x


class FeedForwardBlock(nn.Module):
    """
    Feed-forward block of the Transformer, a simple MLP
    
    Parameters:
    - in_dim: Dimension of the input
    - exp_fct: Expansion factor, how much to expand hidden layer
    - dropout: Dropout rate
    """

    def __init__(self, in_dim, exp_fct=4, dropout=0.1):
        super(FeedForwardBlock, self).__init__()

        hidden_dim = in_dim * exp_fct

        self.linear_in = nn.Linear(in_dim, hidden_dim)
        self.silu = nn.SiLU()
        self.dropout = nn.Dropout(dropout)
        self.linear_out = nn.Linear(hidden_dim, in_dim)

    def forward(self, x):
        x = self.linear_in(x)
        x = self.silu(x) 
        x = self.dropout(x)
        x = self.linear_out(x)
        return x


class AttentionWrapper(nn.Module):
    """
    Helper to wrap MultiheadAttention because MHA returns tuples
    but ResidualConnection expects a single tensor.
    """
    def __init__(self, emb_dim, n_heads, dropout):
        super(AttentionWrapper, self).__init__()
        self.mha = nn.MultiheadAttention(emb_dim, n_heads, dropout, batch_first=True)

    def forward(self, x):
        # Only return the output tensor, not the attention weights
        out, _ = self.mha(x, x, x)
        return out


class EncoderBlock(nn.Module):
    """
    Encoder-Only Transformer to capture global relationships using multihead attention
    
    Parameters:
    - emb_dim: Embedding dimension
    - n_heads: Number of heads for MHA
    - dropout: Dropout rate, 0.1 similar to Vaswani et al.
    - expansion: Expansion rate
    """

    def __init__(self, emb_dim, n_heads, dropout=0.1, expansion=4):
        super(EncoderBlock, self).__init__()

        self.attention = AttentionWrapper(emb_dim, n_heads, dropout)
        self.ffn = FeedForwardBlock(emb_dim, expansion, dropout)
        self.residual1 = ResidualConnection(self.attention, emb_dim, dropout)
        self.residual2 = ResidualConnection(self.ffn, emb_dim, dropout)

    def forward(self, x, mask=None):
        x = self.residual1(x)
        x = self.residual2(x)
        return x


class MLPClassifier(nn.Module):
    """
    Final MLP Classifier Layer
    
    Parameters:
    - tr_out_dim: Dimension of transformer output (emb_dim)
    - dropout: Dropout rate
    - pooling_type: Pooling type, here 'avg'
    """

    def __init__(self, tr_out_dim, dropout=0.5, pooling_type='avg'):
        super(MLPClassifier, self).__init__()
        self.pooling_type = pooling_type
        input_dim = tr_out_dim

        self.linear_in = nn.Linear(input_dim, tr_out_dim // 2)
        self.linear_out = nn.Linear(tr_out_dim // 2, 1)
        self.elu = nn.ELU(True)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = torch.mean(x, dim=1) # (batch, emb_dim)
        x = self.linear_in(x)
        x = self.elu(x)
        x = self.dropout(x)
        x = self.linear_out(x)

        return x


# ====================
# Main CNN-Transformer
# ====================

class EEGClassifier(nn.Module):
    """
    Main CNN-Transformer Model

    Parameters:
    - n_channels: Nmber of EEG input channels (64)
    - emb_dim: Dimension of embeddings
    - max_patches: Max length of time sequences after CNN pooling
    - n_heads: Number of heads for multihead attention
    - dropout: Dropout rate
    - fs: Sampling rate of EEG data, for PhysioNet dataset 160 Hz
    """

    def __init__(self, n_channels=64, emb_dim=128, max_patches=500, n_heads=4, dropout=0.1, fs=160):
        super(EEGClassifier, self).__init__()
    
        self.cnn = CNN(n_channels=n_channels, emb_dim=emb_dim, fs=fs)
        self.positional_encoding = PositionalEncoding(emb_dim, max_patches) 
        self.transformer = EncoderBlock(emb_dim, n_heads, dropout)
        self.mlp = MLPClassifier(tr_out_dim=emb_dim, pooling_type='concat') # Assuming pooling reduces patches by half

    def forward(self, x):
        """
        Args: 
            x: Input EEG data (batch, n_channels, n_time_points)
        Returns: 
            logits for binary classification (batch size,)
        """
        x = self.cnn(x) # (batch, n_channels, n_samples) -> (batch, time_reduced, emb_dim)
        x = self.positional_encoding(x) # (batch, time_reduced, emb_dim)
        x = self.transformer(x) # (batch, time_reduced, emb_dim)
        x = self.mlp(x) # (batch, time_reduced, emb_dim) -> (batch, 1)
        
        return x.squeeze(-1) # (batch, 1) -> (batch,)