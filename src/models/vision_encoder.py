"""Vision encoder using EfficientNet or MobileViT."""
import torch
import torch.nn as nn
import torchvision.models as models
from typing import Optional


class VisionEncoder(nn.Module):
    """Vision encoder for extracting features from video frames."""
    
    def __init__(
        self,
        model_name: str = "efficientnet_b0",
        pretrained: bool = True,
        output_dim: int = 512,
        dropout: float = 0.2,
        freeze_backbone: bool = False
    ):
        """
        Initialize vision encoder.
        
        Args:
            model_name: Backbone architecture name
            pretrained: Whether to use pretrained weights
            output_dim: Output feature dimension
            dropout: Dropout rate
            freeze_backbone: Whether to freeze backbone weights
        """
        super().__init__()
        
        self.model_name = model_name
        self.output_dim = output_dim
        
        # Load backbone
        if model_name == "efficientnet_b0":
            weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
            self.backbone = models.efficientnet_b0(weights=weights)
            backbone_out_dim = 1280
            # Remove classifier
            self.backbone.classifier = nn.Identity()
            
        elif model_name == "mobilenet_v3_small":
            weights = models.MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
            self.backbone = models.mobilenet_v3_small(weights=weights)
            backbone_out_dim = 576
            # Remove classifier
            self.backbone.classifier = nn.Identity()
            
        elif model_name == "resnet50":
            weights = models.ResNet50_Weights.DEFAULT if pretrained else None
            self.backbone = models.resnet50(weights=weights)
            backbone_out_dim = 2048
            # Remove classifier
            self.backbone.fc = nn.Identity()
            
        else:
            raise ValueError(f"Unknown model: {model_name}")
        
        # Freeze backbone if requested
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
        
        # Projection head
        self.projection = nn.Sequential(
            nn.Linear(backbone_out_dim, output_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(output_dim, output_dim)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (B, T, C, H, W)
            
        Returns:
            Features of shape (B, T, output_dim)
        """
        batch_size, num_frames, C, H, W = x.shape
        
        # Reshape to process all frames
        x = x.view(batch_size * num_frames, C, H, W)
        
        # Extract features
        features = self.backbone(x)
        
        # Project to output dimension
        features = self.projection(features)
        
        # Reshape back to (B, T, D)
        features = features.view(batch_size, num_frames, self.output_dim)
        
        return features
    
    def get_feature_dim(self) -> int:
        """Return output feature dimension."""
        return self.output_dim


class SpatialAttention(nn.Module):
    """Spatial attention mechanism for frame-level features."""
    
    def __init__(self, feature_dim: int):
        """
        Initialize spatial attention.
        
        Args:
            feature_dim: Input feature dimension
        """
        super().__init__()
        
        self.attention = nn.Sequential(
            nn.Linear(feature_dim, feature_dim // 2),
            nn.Tanh(),
            nn.Linear(feature_dim // 2, 1),
            nn.Softmax(dim=1)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply spatial attention.
        
        Args:
            x: Input features (B, T, D)
            
        Returns:
            Attended features (B, D)
        """
        # Compute attention weights
        weights = self.attention(x)  # (B, T, 1)
        
        # Apply attention
        attended = torch.sum(x * weights, dim=1)  # (B, D)
        
        return attended
