#!/usr/bin/env python3
"""
PyTorch Vision Transformer Implementation for Violence Detection
===============================================================
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
import math


class PatchEmbedding(nn.Module):
    """
    Video patch embedding using 3D convolution.
    Converts video frames into patches and applies linear projection.
    """
    def __init__(self, img_size=224, patch_size=16, in_channels=3, embed_dim=256, sequence_length=16):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.sequence_length = sequence_length
        
        # Temporal stride and kernel size for Conv3D
        temporal_kernel_size = 2
        temporal_stride = 2

        # Calculate number of patches
        self.num_patches_per_frame = (img_size // patch_size) ** 2
        # Adjust num_patches based on the temporal downsampling from Conv3D
        self.num_patches = (sequence_length // temporal_stride) * self.num_patches_per_frame
        
        # 3D Convolution for tubelet embedding
        self.projection = nn.Conv3d(
            in_channels=in_channels,
            out_channels=embed_dim,
            kernel_size=(temporal_kernel_size, patch_size, patch_size),
            stride=(temporal_stride, patch_size, patch_size),
            padding=0
        )
        
    def forward(self, x):
        # x shape: (batch_size, sequence_length, channels, height, width)
        # Rearrange to (batch_size, channels, sequence_length, height, width)
        x = rearrange(x, 'b t c h w -> b c t h w')
        
        # Apply 3D convolution
        x = self.projection(x)  # (batch_size, embed_dim, t', h', w')
        
        # Flatten spatial and temporal dimensions
        x = rearrange(x, 'b e t h w -> b (t h w) e')
        
        return x


class MultiHeadAttention(nn.Module):
    """Multi-Head Self-Attention mechanism."""
    
    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super().__init__()
        assert embed_dim % num_heads == 0
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        self.qkv = nn.Linear(embed_dim, embed_dim * 3)
        self.attn_dropout = nn.Dropout(dropout)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.proj_dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        B, N, C = x.shape
        
        # Generate Q, K, V
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # Attention computation
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_dropout(attn)
        
        # Apply attention to values
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_dropout(x)
        
        return x


class MLP(nn.Module):
    """Multi-Layer Perceptron (Feed Forward Network)."""
    
    def __init__(self, embed_dim, hidden_dim, dropout=0.1):
        super().__init__()
        self.fc1 = nn.Linear(embed_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.GELU()
        
    def forward(self, x):
        x = self.fc1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class TransformerBlock(nn.Module):
    """Transformer Encoder Block."""
    
    def __init__(self, embed_dim, num_heads, mlp_ratio=4.0, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadAttention(embed_dim, num_heads, dropout)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = MLP(embed_dim, int(embed_dim * mlp_ratio), dropout)
        
    def forward(self, x):
        # Self-attention with residual connection
        x = x + self.attn(self.norm1(x))
        # MLP with residual connection
        x = x + self.mlp(self.norm2(x))
        return x


class VisionTransformer(nn.Module):
    """
    Vision Transformer for Video Violence Detection.
    
    Args:
        img_size: Input image size
        patch_size: Patch size for embedding
        in_channels: Number of input channels
        embed_dim: Embedding dimension
        num_heads: Number of attention heads
        num_layers: Number of transformer layers
        num_classes: Number of output classes (1 for binary classification)
        sequence_length: Number of frames in video sequence
        dropout: Dropout rate
    """
    
    def __init__(
        self,
        img_size=224,
        patch_size=16,
        in_channels=3,
        embed_dim=256,
        num_heads=8,
        num_layers=6,
        num_classes=1,
        sequence_length=16,
        dropout=0.1
    ):
        super().__init__()
        
        self.embed_dim = embed_dim
        self.num_classes = num_classes
        
        # Patch embedding
        self.patch_embed = PatchEmbedding(
            img_size=img_size,
            patch_size=patch_size,
            in_channels=in_channels,
            embed_dim=embed_dim,
            sequence_length=sequence_length
        )
        
        # Class token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        
        # Positional embedding
        self.pos_embed = nn.Parameter(
            torch.zeros(1, self.patch_embed.num_patches + 1, embed_dim)
        )
        
        self.pos_dropout = nn.Dropout(dropout)
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, dropout=dropout)
            for _ in range(num_layers)
        ])
        
        # Final layer norm
        self.norm = nn.LayerNorm(embed_dim)
        
        # Classification head
        self.head = nn.Linear(embed_dim, num_classes)
        
        # Initialize weights
        self._init_weights()
        
    def _init_weights(self):
        """Initialize model weights."""
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        # x shape: (batch_size, sequence_length, channels, height, width)
        B = x.shape[0]
        
        # Patch embedding
        x = self.patch_embed(x)  # (B, num_patches, embed_dim)
        
        # Add class token
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)  # (B, num_patches + 1, embed_dim)
        
        # Add positional embedding
        x = x + self.pos_embed
        x = self.pos_dropout(x)
        
        # Apply transformer blocks
        for block in self.blocks:
            x = block(x)
        
        # Final layer norm
        x = self.norm(x)
        
        # Classification head (use class token)
        x = self.head(x[:, 0])  # (B, num_classes)
        
        return x


def create_vit_model(
    img_size=224,
    patch_size=16,
    embed_dim=256,
    num_heads=8,
    num_layers=6,
    sequence_length=16,
    dropout=0.1
):
    """Create a Vision Transformer model with specified parameters."""
    model = VisionTransformer(
        img_size=img_size,
        patch_size=patch_size,
        embed_dim=embed_dim,
        num_heads=num_heads,
        num_layers=num_layers,
        sequence_length=sequence_length,
        dropout=dropout
    )
    return model


if __name__ == "__main__":
    # Test the model
    device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    model = create_vit_model().to(device)
    
    # Test input: (batch_size, sequence_length, channels, height, width)
    test_input = torch.randn(2, 16, 3, 224, 224).to(device)
    
    with torch.no_grad():
        output = model(test_input)
        print(f"Input shape: {test_input.shape}")
        print(f"Output shape: {output.shape}")
        print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
