"""Test model architectures."""
import pytest
import torch

from src.models.vision_encoder import VisionEncoder
from src.models.audio_encoder import AudioEncoder
from src.models.fusion import MultiModalFusion
from src.models.multimodal_model import MultiModalVideoModel


class TestVisionEncoder:
    """Test vision encoder."""
    
    def test_vision_encoder_forward(self):
        """Test vision encoder forward pass."""
        encoder = VisionEncoder(
            model_name='efficientnet_b0',
            pretrained=False,
            output_dim=512
        )
        
        # Create dummy input
        x = torch.randn(2, 16, 3, 224, 224)
        
        # Forward pass
        features = encoder(x)
        
        assert features.shape == (2, 16, 512)


class TestAudioEncoder:
    """Test audio encoder."""
    
    def test_audio_encoder_forward(self):
        """Test audio encoder forward pass."""
        encoder = AudioEncoder(
            pretrained=False,
            output_dim=512
        )
        
        # Create dummy audio
        audio = torch.randn(2, 16000)
        
        # Forward pass
        features = encoder(audio)
        
        assert features.shape == (2, 512)


class TestMultiModalFusion:
    """Test multi-modal fusion."""
    
    def test_fusion_forward(self):
        """Test fusion forward pass."""
        fusion = MultiModalFusion(
            hidden_dim=512,
            fusion_type='cross_attention'
        )
        
        # Create dummy features
        video_features = torch.randn(2, 16, 512)
        audio_features = torch.randn(2, 512)
        
        # Forward pass
        fused = fusion(video_features, audio_features)
        
        assert fused.shape == (2, 16, 512)


class TestMultiModalModel:
    """Test complete multi-modal model."""
    
    def test_model_forward(self):
        """Test model forward pass."""
        config = {
            'vision_encoder': {
                'name': 'efficientnet_b0',
                'pretrained': False,
                'output_dim': 512
            },
            'audio_encoder': {
                'pretrained': False,
                'output_dim': 512
            },
            'temporal_encoder': {
                'type': 'transformer',
                'hidden_dim': 512,
                'num_layers': 2
            },
            'fusion': {
                'type': 'cross_attention',
                'hidden_dim': 512
            },
            'classification_head': {
                'num_classes': 400
            },
            'captioning_head': {}
        }
        
        model = MultiModalVideoModel(config)
        
        # Create dummy inputs
        video = torch.randn(2, 16, 3, 224, 224)
        audio = torch.randn(2, 16000)
        
        # Forward pass
        outputs = model(video, audio)
        
        assert 'class_logits' in outputs
        assert outputs['class_logits'].shape == (2, 400)


if __name__ == '__main__':
    pytest.main([__file__])
