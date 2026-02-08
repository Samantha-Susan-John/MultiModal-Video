"""Data augmentation for multi-modal video data."""
import torch
import torchvision.transforms as T
import torchaudio.transforms as AT
import random
from typing import Optional, Tuple


class MultiModalAugmentation:
    """Augmentation pipeline for video and audio data."""
    
    def __init__(
        self,
        mode: str = 'train',
        video_size: Tuple[int, int] = (224, 224),
        audio_augment: bool = True,
        temporal_augment: bool = True
    ):
        """
        Initialize augmentation pipeline.
        
        Args:
            mode: 'train' or 'val'
            video_size: Target video frame size
            audio_augment: Apply audio augmentations
            temporal_augment: Apply temporal augmentations
        """
        self.mode = mode
        self.video_size = video_size
        self.audio_augment = audio_augment and (mode == 'train')
        self.temporal_augment = temporal_augment and (mode == 'train')
        
        # Video transforms
        if mode == 'train':
            self.video_transform = T.Compose([
                T.RandomResizedCrop(video_size, scale=(0.8, 1.0)),
                T.RandomHorizontalFlip(p=0.5),
                T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        else:
            self.video_transform = T.Compose([
                T.Resize(video_size),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        
        # Audio transforms
        if self.audio_augment:
            self.noise_level = 0.005
            self.time_stretch_rate = (0.8, 1.2)
            self.pitch_shift_semitones = 2
    
    def __call__(
        self,
        video_frames: torch.Tensor,
        audio_waveform: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Apply augmentations.
        
        Args:
            video_frames: Video tensor of shape (T, C, H, W)
            audio_waveform: Audio tensor of shape (channels, samples)
            
        Returns:
            Augmented (video, audio) tuple
        """
        # Video augmentation
        video_frames = self._augment_video(video_frames)
        
        # Audio augmentation
        if audio_waveform is not None:
            audio_waveform = self._augment_audio(audio_waveform)
        
        return video_frames, audio_waveform
    
    def _augment_video(self, frames: torch.Tensor) -> torch.Tensor:
        """Apply video augmentations."""
        T, C, H, W = frames.shape
        
        # Apply spatial transforms to each frame
        augmented_frames = []
        for i in range(T):
            frame = frames[i]  # (C, H, W)
            frame = self.video_transform(frame)
            augmented_frames.append(frame)
        
        frames = torch.stack(augmented_frames, dim=0)
        
        # Temporal augmentation
        if self.temporal_augment:
            frames = self._temporal_augment(frames)
        
        return frames
    
    def _temporal_augment(self, frames: torch.Tensor) -> torch.Tensor:
        """Apply temporal augmentations (random temporal cropping, etc.)."""
        T = frames.shape[0]
        
        # Random temporal crop
        if T > 16 and random.random() > 0.5:
            start_idx = random.randint(0, T - 16)
            frames = frames[start_idx:start_idx + 16]
        
        # Random frame skip (temporal downsampling)
        if random.random() > 0.7:
            indices = torch.arange(0, frames.shape[0], 2)
            frames = frames[indices]
        
        return frames
    
    def _augment_audio(self, waveform: torch.Tensor) -> torch.Tensor:
        """Apply audio augmentations."""
        if not self.audio_augment:
            return waveform
        
        # Add random noise
        if random.random() > 0.5:
            noise = torch.randn_like(waveform) * self.noise_level
            waveform = waveform + noise
        
        # Random gain
        if random.random() > 0.5:
            gain = random.uniform(0.8, 1.2)
            waveform = waveform * gain
        
        # Clip to prevent overflow
        waveform = torch.clamp(waveform, -1.0, 1.0)
        
        return waveform


class VideoAugmentation:
    """Video-only augmentation pipeline."""
    
    def __init__(self, mode: str = 'train', size: Tuple[int, int] = (224, 224)):
        """Initialize video augmentation."""
        if mode == 'train':
            self.transform = T.Compose([
                T.RandomResizedCrop(size, scale=(0.8, 1.0)),
                T.RandomHorizontalFlip(),
                T.ColorJitter(brightness=0.2, contrast=0.2),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        else:
            self.transform = T.Compose([
                T.Resize(size),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
    
    def __call__(self, frames: torch.Tensor) -> torch.Tensor:
        """Apply transforms to all frames."""
        augmented = []
        for i in range(frames.shape[0]):
            augmented.append(self.transform(frames[i]))
        return torch.stack(augmented, dim=0)


class AudioAugmentation:
    """Audio-only augmentation pipeline."""
    
    def __init__(self, mode: str = 'train', sample_rate: int = 16000):
        """Initialize audio augmentation."""
        self.mode = mode
        self.sample_rate = sample_rate
        self.noise_level = 0.005 if mode == 'train' else 0.0
    
    def __call__(self, waveform: torch.Tensor) -> torch.Tensor:
        """Apply audio augmentations."""
        if self.mode != 'train':
            return waveform
        
        # Add noise
        if random.random() > 0.5:
            noise = torch.randn_like(waveform) * self.noise_level
            waveform = waveform + noise
        
        # Random gain
        if random.random() > 0.5:
            gain = random.uniform(0.8, 1.2)
            waveform = waveform * gain
        
        return torch.clamp(waveform, -1.0, 1.0)
