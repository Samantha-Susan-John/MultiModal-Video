"""Audio encoder using pretrained Wav2Vec2."""
import torch
import torch.nn as nn
from transformers import Wav2Vec2Model
from typing import Optional


class AudioEncoder(nn.Module):
    """Audio encoder using pretrained Wav2Vec2."""
    
    def __init__(
        self,
        model_name: str = "facebook/wav2vec2-base",
        pretrained: bool = True,
        output_dim: int = 512,
        freeze_layers: int = 8,
        pooling: str = "mean"
    ):
        """
        Initialize audio encoder.
        
        Args:
            model_name: Pretrained model name
            pretrained: Whether to use pretrained weights
            output_dim: Output feature dimension
            freeze_layers: Number of transformer layers to freeze
            pooling: Pooling strategy (mean, max, attention)
        """
        super().__init__()
        
        self.pooling = pooling
        self.output_dim = output_dim
        
        # Load pretrained Wav2Vec2
        if pretrained:
            self.wav2vec = Wav2Vec2Model.from_pretrained(model_name)
        else:
            from transformers import Wav2Vec2Config
            config = Wav2Vec2Config.from_pretrained(model_name)
            self.wav2vec = Wav2Vec2Model(config)
        
        # Freeze early layers
        if freeze_layers > 0:
            # Freeze feature extractor
            for param in self.wav2vec.feature_extractor.parameters():
                param.requires_grad = False
            
            # Freeze specified transformer layers
            for i, layer in enumerate(self.wav2vec.encoder.layers):
                if i < freeze_layers:
                    for param in layer.parameters():
                        param.requires_grad = False
        
        # Get hidden dimension
        hidden_dim = self.wav2vec.config.hidden_size
        
        # Attention pooling
        if pooling == "attention":
            self.attention = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.Tanh(),
                nn.Linear(hidden_dim // 2, 1)
            )
        
        # Projection layer
        self.projection = nn.Sequential(
            nn.Linear(hidden_dim, output_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(output_dim, output_dim)
        )
    
    def forward(
        self, 
        audio: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            audio: Audio waveform tensor (B, seq_len)
            attention_mask: Optional attention mask
            
        Returns:
            Audio features of shape (B, output_dim)
        """
        # Extract features
        outputs = self.wav2vec(
            audio,
            attention_mask=attention_mask,
            output_hidden_states=True
        )
        
        # Get hidden states
        hidden_states = outputs.last_hidden_state  # (B, T, hidden_dim)
        
        # Apply pooling
        if self.pooling == "mean":
            if attention_mask is not None:
                # Masked mean pooling
                mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_states.size())
                sum_hidden = torch.sum(hidden_states * mask_expanded, dim=1)
                sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
                pooled = sum_hidden / sum_mask
            else:
                pooled = torch.mean(hidden_states, dim=1)
                
        elif self.pooling == "max":
            pooled = torch.max(hidden_states, dim=1)[0]
            
        elif self.pooling == "attention":
            # Attention-based pooling
            attention_weights = self.attention(hidden_states)  # (B, T, 1)
            attention_weights = torch.softmax(attention_weights, dim=1)
            pooled = torch.sum(hidden_states * attention_weights, dim=1)
            
        else:
            raise ValueError(f"Unknown pooling: {self.pooling}")
        
        # Project to output dimension
        features = self.projection(pooled)
        
        return features
    
    def get_feature_dim(self) -> int:
        """Return output feature dimension."""
        return self.output_dim


class AudioVisualSyncDetector(nn.Module):
    """Detect audio-visual synchronization."""
    
    def __init__(self, feature_dim: int):
        """
        Initialize sync detector.
        
        Args:
            feature_dim: Input feature dimension
        """
        super().__init__()
        
        self.detector = nn.Sequential(
            nn.Linear(feature_dim * 2, feature_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(feature_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
    
    def forward(
        self, 
        video_features: torch.Tensor,
        audio_features: torch.Tensor
    ) -> torch.Tensor:
        """
        Predict if audio and video are synchronized.
        
        Args:
            video_features: Video features (B, D)
            audio_features: Audio features (B, D)
            
        Returns:
            Sync probability (B, 1)
        """
        # Concatenate features
        combined = torch.cat([video_features, audio_features], dim=-1)
        
        # Predict sync
        sync_prob = self.detector(combined)
        
        return sync_prob
