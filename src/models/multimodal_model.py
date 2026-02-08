"""Complete multi-modal video understanding model."""
import torch
import torch.nn as nn
from typing import Dict, Optional, Tuple

from .vision_encoder import VisionEncoder
from .audio_encoder import AudioEncoder, AudioVisualSyncDetector
from .fusion import TemporalTransformer, TemporalLSTM, MultiModalFusion
from .task_heads import ClassificationHead, CaptioningHead


class MultiModalVideoModel(nn.Module):
    """Complete multi-modal video understanding model."""
    
    def __init__(self, config: Dict):
        """
        Initialize multi-modal model.
        
        Args:
            config: Model configuration dictionary
        """
        super().__init__()
        
        self.config = config
        
        # Vision encoder
        vision_config = config.get('vision_encoder', {})
        self.vision_encoder = VisionEncoder(
            model_name=vision_config.get('name', 'efficientnet_b0'),
            pretrained=vision_config.get('pretrained', True),
            output_dim=vision_config.get('output_dim', 512),
            dropout=vision_config.get('dropout', 0.2),
            freeze_backbone=vision_config.get('freeze_backbone', False)
        )
        
        # Audio encoder
        audio_config = config.get('audio_encoder', {})
        self.audio_encoder = AudioEncoder(
            model_name=audio_config.get('model_name', 'facebook/wav2vec2-base'),
            pretrained=audio_config.get('pretrained', True),
            output_dim=audio_config.get('output_dim', 512),
            freeze_layers=audio_config.get('freeze_layers', 8),
            pooling=audio_config.get('pooling', 'mean')
        )
        
        # Temporal encoder
        temporal_config = config.get('temporal_encoder', {})
        temporal_type = temporal_config.get('type', 'transformer')
        temporal_hidden = temporal_config.get('hidden_dim', 512)
        
        # Add projection layer if vision and temporal dims don't match
        vision_out_dim = vision_config.get('output_dim', 512)
        if vision_out_dim != temporal_hidden:
            self.vision_projection = nn.Linear(vision_out_dim, temporal_hidden)
        else:
            self.vision_projection = None
        
        if temporal_type == 'transformer':
            self.temporal_encoder = TemporalTransformer(
                hidden_dim=temporal_hidden,
                num_layers=temporal_config.get('num_layers', 4),
                num_heads=temporal_config.get('num_heads', 8),
                dropout=temporal_config.get('dropout', 0.1),
                max_seq_length=temporal_config.get('max_seq_length', 32)
            )
        else:
            self.temporal_encoder = TemporalLSTM(
                input_dim=temporal_hidden,
                hidden_dim=temporal_hidden,
                num_layers=temporal_config.get('num_layers', 2),
                dropout=temporal_config.get('dropout', 0.1)
            )
        
        # Multi-modal fusion
        fusion_config = config.get('fusion', {})
        fusion_hidden = fusion_config.get('hidden_dim', 512)
        
        # Add projection layers if audio and fusion dims don't match
        audio_out_dim = audio_config.get('output_dim', 512)
        if audio_out_dim != fusion_hidden:
            self.audio_projection = nn.Linear(audio_out_dim, fusion_hidden)
        else:
            self.audio_projection = None
        
        # Add projection from temporal to fusion if needed
        if temporal_hidden != fusion_hidden:
            self.temporal_to_fusion = nn.Linear(temporal_hidden, fusion_hidden)
        else:
            self.temporal_to_fusion = None
        
        self.fusion = MultiModalFusion(
            hidden_dim=fusion_hidden,
            fusion_type=fusion_config.get('type', 'cross_attention'),
            num_heads=fusion_config.get('num_heads', 8),
            dropout=fusion_config.get('dropout', 0.1)
        )
        
        # Task heads
        classification_config = config.get('classification_head', {})
        self.classification_head = ClassificationHead(
            input_dim=fusion_hidden,
            hidden_dims=classification_config.get('hidden_dims', [512, 256]),
            num_classes=classification_config.get('num_classes', 400),
            dropout=classification_config.get('dropout', 0.3)
        )
        
        captioning_config = config.get('captioning_head', {})
        self.captioning_head = CaptioningHead(
            input_dim=fusion_hidden,
            hidden_dim=captioning_config.get('hidden_dim', 512),
            num_layers=captioning_config.get('num_layers', 2),
            dropout=captioning_config.get('dropout', 0.2)
        )
        
        # Audio-visual sync detector
        self.sync_detector = AudioVisualSyncDetector(
            feature_dim=fusion_config.get('hidden_dim', 512)
        )
    
    def forward(
        self,
        video: torch.Tensor,
        audio: Optional[torch.Tensor] = None,
        captions: Optional[torch.Tensor] = None,
        return_features: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            video: Video frames (B, T, C, H, W)
            audio: Audio waveform (B, audio_len) or features (B, D)
            captions: Target captions for training (B, seq_len)
            return_features: Whether to return intermediate features
            
        Returns:
            Dictionary containing model outputs
        """
        outputs = {}
        
        # Extract vision features
        video_features = self.vision_encoder(video)  # (B, T, D)
        
        # Project to temporal dimension if needed
        if self.vision_projection is not None:
            B, T, D = video_features.shape
            video_features = video_features.view(B * T, D)
            video_features = self.vision_projection(video_features)
            video_features = video_features.view(B, T, -1)
        
        # Temporal modeling
        video_temporal = self.temporal_encoder(video_features)  # (B, T, D)
        
        # Project temporal features to fusion dimension if needed
        if self.temporal_to_fusion is not None:
            B, T, D = video_temporal.shape
            video_temporal = video_temporal.view(B * T, D)
            video_temporal = self.temporal_to_fusion(video_temporal)
            video_temporal = video_temporal.view(B, T, -1)
        
        # Extract audio features if provided
        if audio is not None:
            if audio.dim() == 2 and audio.size(-1) > 1000:
                # Raw audio waveform
                audio_features = self.audio_encoder(audio)  # (B, D)
            else:
                # Already extracted features
                audio_features = audio
            
            # Project audio to fusion dimension if needed
            if self.audio_projection is not None:
                audio_features = self.audio_projection(audio_features)
            
            # Multi-modal fusion
            fused_features = self.fusion(video_temporal, audio_features)  # (B, T, D)
            
            # Compute sync score
            video_pooled = torch.mean(video_temporal, dim=1)
            sync_score = self.sync_detector(video_pooled, audio_features)
            outputs['sync_score'] = sync_score
            
        else:
            # Video-only mode
            fused_features = video_temporal
            audio_features = None
        
        # Classification
        class_logits = self.classification_head(fused_features)
        outputs['class_logits'] = class_logits
        
        # Captioning
        if captions is not None:
            caption_logits = self.captioning_head(fused_features, captions)
            outputs['caption_logits'] = caption_logits
        
        # Return features if requested
        if return_features:
            outputs['video_features'] = video_features
            outputs['video_temporal'] = video_temporal
            outputs['audio_features'] = audio_features
            outputs['fused_features'] = fused_features
        
        return outputs
    
    def generate_caption(
        self,
        video: torch.Tensor,
        audio: Optional[torch.Tensor] = None,
        max_length: int = 50
    ) -> torch.Tensor:
        """
        Generate video captions.
        
        Args:
            video: Video frames (B, T, C, H, W)
            audio: Optional audio features
            max_length: Maximum caption length
            
        Returns:
            Generated caption tokens (B, seq_len)
        """
        self.eval()
        with torch.no_grad():
            # Extract and fuse features
            video_features = self.vision_encoder(video)
            video_temporal = self.temporal_encoder(video_features)
            
            if audio is not None:
                audio_features = self.audio_encoder(audio)
                fused_features = self.fusion(video_temporal, audio_features)
            else:
                fused_features = video_temporal
            
            # Generate caption
            captions = self.captioning_head.generate(
                fused_features,
                max_length=max_length
            )
        
        return captions
    
    def get_num_parameters(self) -> Tuple[int, int]:
        """
        Get number of parameters.
        
        Returns:
            Tuple of (total params, trainable params)
        """
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return total, trainable
