"""Test data pipeline."""
import pytest
import torch
import numpy as np
from pathlib import Path

from src.data.video_loader import VideoLoader, extract_uniform_frames
from src.data.augmentation import VideoAugmentation, TemporalAugmentation


class TestVideoLoader:
    """Test video loading functionality."""
    
    def test_video_loader_initialization(self):
        """Test VideoLoader initialization."""
        loader = VideoLoader(target_fps=30, max_frames=300)
        assert loader.target_fps == 30
        assert loader.max_frames == 300
    
    def test_uniform_frame_extraction(self):
        """Test uniform frame extraction."""
        # Create dummy frames
        frames = np.random.randint(0, 255, (100, 224, 224, 3), dtype=np.uint8)
        
        # Extract 32 frames
        sampled = extract_uniform_frames(frames, num_frames=32)
        
        assert sampled.shape == (32, 224, 224, 3)
        assert sampled.dtype == np.uint8


class TestAugmentation:
    """Test augmentation functionality."""
    
    def test_video_augmentation(self):
        """Test video augmentation."""
        aug = VideoAugmentation(image_size=(224, 224), is_training=True)
        
        # Create dummy frames
        frames = np.random.randint(0, 255, (16, 224, 224, 3), dtype=np.uint8)
        
        # Apply augmentation
        augmented = aug(frames)
        
        assert isinstance(augmented, torch.Tensor)
        assert augmented.shape == (16, 3, 224, 224)
    
    def test_temporal_augmentation(self):
        """Test temporal augmentation."""
        frames = np.random.randint(0, 255, (100, 224, 224, 3), dtype=np.uint8)
        
        # Random temporal crop
        cropped = TemporalAugmentation.random_temporal_crop(frames, crop_size=32)
        
        assert cropped.shape[0] == 32
        assert cropped.shape[1:] == frames.shape[1:]


if __name__ == '__main__':
    pytest.main([__file__])
