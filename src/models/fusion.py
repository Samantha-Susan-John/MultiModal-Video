"""Temporal modeling and multi-modal fusion."""
import torch
import torch.nn as nn
import math
from typing import Optional


class TemporalTransformer(nn.Module):
    """Transformer encoder for temporal modeling."""
    
    def __init__(
        self,
        hidden_dim: int = 512,
        num_layers: int = 4,
        num_heads: int = 8,
        dropout: float = 0.1,
        max_seq_length: int = 32
    ):
        """
        Initialize temporal transformer.
        
        Args:
            hidden_dim: Hidden dimension
            num_layers: Number of transformer layers
            num_heads: Number of attention heads
            dropout: Dropout rate
            max_seq_length: Maximum sequence length
        """
        super().__init__()
        
        self.hidden_dim = hidden_dim
        
        # Positional encoding
        self.pos_encoding = PositionalEncoding(hidden_dim, max_seq_length)
        
        # Transformer encoder layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )
        
        self.norm = nn.LayerNorm(hidden_dim)
    
    def forward(
        self, 
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input features (B, T, D)
            mask: Optional attention mask
            
        Returns:
            Temporal features (B, T, D)
        """
        # Add positional encoding
        x = self.pos_encoding(x)
        
        # Apply transformer
        x = self.transformer(x, src_key_padding_mask=mask)
        
        # Layer norm
        x = self.norm(x)
        
        return x


class TemporalLSTM(nn.Module):
    """LSTM for temporal modeling."""
    
    def __init__(
        self,
        input_dim: int = 512,
        hidden_dim: int = 512,
        num_layers: int = 2,
        dropout: float = 0.1,
        bidirectional: bool = True
    ):
        """
        Initialize temporal LSTM.
        
        Args:
            input_dim: Input feature dimension
            hidden_dim: Hidden dimension
            num_layers: Number of LSTM layers
            dropout: Dropout rate
            bidirectional: Whether to use bidirectional LSTM
        """
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.bidirectional = bidirectional
        
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional,
            batch_first=True
        )
        
        # Project back to hidden_dim if bidirectional
        if bidirectional:
            self.projection = nn.Linear(hidden_dim * 2, hidden_dim)
        else:
            self.projection = None
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input features (B, T, D)
            
        Returns:
            Temporal features (B, T, D)
        """
        # Apply LSTM
        output, (hidden, cell) = self.lstm(x)
        
        # Project if bidirectional
        if self.projection is not None:
            output = self.projection(output)
        
        return output


class CrossModalAttention(nn.Module):
    """Cross-modal attention fusion."""
    
    def __init__(
        self,
        hidden_dim: int = 512,
        num_heads: int = 8,
        dropout: float = 0.1
    ):
        """
        Initialize cross-modal attention.
        
        Args:
            hidden_dim: Hidden dimension
            num_heads: Number of attention heads
            dropout: Dropout rate
        """
        super().__init__()
        
        self.multihead_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.Dropout(dropout)
        )
    
    def forward(
        self,
        query: torch.Tensor,
        key_value: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            query: Query features (B, T_q, D)
            key_value: Key/Value features (B, T_kv, D)
            
        Returns:
            Fused features (B, T_q, D)
        """
        # Cross-attention
        attn_output, _ = self.multihead_attn(
            query=query,
            key=key_value,
            value=key_value
        )
        
        # Residual connection and norm
        x = self.norm1(query + attn_output)
        
        # Feed-forward
        ffn_output = self.ffn(x)
        
        # Residual connection and norm
        x = self.norm2(x + ffn_output)
        
        return x


class MultiModalFusion(nn.Module):
    """Multi-modal fusion module."""
    
    def __init__(
        self,
        hidden_dim: int = 512,
        fusion_type: str = "cross_attention",
        num_heads: int = 8,
        dropout: float = 0.1
    ):
        """
        Initialize fusion module.
        
        Args:
            hidden_dim: Hidden dimension
            fusion_type: Fusion strategy (cross_attention, concat, film)
            num_heads: Number of attention heads
            dropout: Dropout rate
        """
        super().__init__()
        
        self.fusion_type = fusion_type
        self.hidden_dim = hidden_dim
        
        if fusion_type == "cross_attention":
            # Bidirectional cross-attention
            self.video_to_audio = CrossModalAttention(hidden_dim, num_heads, dropout)
            self.audio_to_video = CrossModalAttention(hidden_dim, num_heads, dropout)
            
            self.fusion_proj = nn.Sequential(
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout)
            )
            
        elif fusion_type == "concat":
            # Simple concatenation
            self.fusion_proj = nn.Sequential(
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, hidden_dim)
            )
            
        elif fusion_type == "film":
            # FiLM (Feature-wise Linear Modulation)
            self.gamma = nn.Linear(hidden_dim, hidden_dim)
            self.beta = nn.Linear(hidden_dim, hidden_dim)
            
        else:
            raise ValueError(f"Unknown fusion type: {fusion_type}")
    
    def forward(
        self,
        video_features: torch.Tensor,
        audio_features: torch.Tensor
    ) -> torch.Tensor:
        """
        Fuse video and audio features.
        
        Args:
            video_features: Video features (B, T_v, D) or (B, D)
            audio_features: Audio features (B, D)
            
        Returns:
            Fused features (B, T_v, D) or (B, D)
        """
        if self.fusion_type == "cross_attention":
            # Expand audio to match video temporal dimension if needed
            if video_features.dim() == 3 and audio_features.dim() == 2:
                audio_features = audio_features.unsqueeze(1).expand(
                    -1, video_features.size(1), -1
                )
            
            # Bidirectional cross-attention
            video_attended = self.video_to_audio(video_features, audio_features)
            audio_attended = self.audio_to_video(audio_features, video_features)
            
            # Concatenate and project
            fused = torch.cat([video_attended, audio_attended], dim=-1)
            fused = self.fusion_proj(fused)
            
        elif self.fusion_type == "concat":
            # Expand audio if needed
            if video_features.dim() == 3 and audio_features.dim() == 2:
                audio_features = audio_features.unsqueeze(1).expand(
                    -1, video_features.size(1), -1
                )
            
            # Concatenate and project
            fused = torch.cat([video_features, audio_features], dim=-1)
            fused = self.fusion_proj(fused)
            
        elif self.fusion_type == "film":
            # FiLM modulation
            gamma = self.gamma(audio_features)
            beta = self.beta(audio_features)
            
            # Expand if needed
            if video_features.dim() == 3 and gamma.dim() == 2:
                gamma = gamma.unsqueeze(1)
                beta = beta.unsqueeze(1)
            
            fused = gamma * video_features + beta
        
        return fused


class PositionalEncoding(nn.Module):
    """Positional encoding for transformer."""
    
    def __init__(self, d_model: int, max_len: int = 5000):
        """
        Initialize positional encoding.
        
        Args:
            d_model: Model dimension
            max_len: Maximum sequence length
        """
        super().__init__()
        
        # Create positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Add positional encoding.
        
        Args:
            x: Input tensor (B, T, D)
            
        Returns:
            Tensor with positional encoding added
        """
        x = x + self.pe[:, :x.size(1), :]
        return x
