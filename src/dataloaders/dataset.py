"""Dataset classes for video understanding."""
import json
from pathlib import Path
from typing import Dict, Optional, Tuple, List
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np

from .video_loader import VideoLoader, uniform_temporal_sampling
from .audio_extractor import AudioExtractor, pad_or_truncate
from .augmentation import MultiModalAugmentation


class Kinetics400Dataset(Dataset):
    """Dataset for Kinetics-400 video classification."""
    
    def __init__(
        self,
        root_dir: str,
        split: str = 'train',
        config: Optional[Dict] = None,
        transform: Optional[MultiModalAugmentation] = None,
        load_audio: bool = True,
        load_video: bool = True
    ):
        """
        Initialize dataset.
        
        Args:
            root_dir: Root directory containing videos and annotations
            split: 'train', 'val', or 'test'
            config: Data configuration dictionary
            transform: Augmentation pipeline
            load_audio: Whether to load audio
            load_video: Whether to load video frames
        """
        self.root_dir = Path(root_dir)
        self.split = split
        self.config = config or {}
        self.load_audio = load_audio
        self.load_video = load_video
        
        # Load annotations
        annotation_file = self.root_dir / f'{split}_annotations.json'
        if not annotation_file.exists():
            raise FileNotFoundError(f"Annotations not found: {annotation_file}")
        
        with open(annotation_file, 'r') as f:
            self.annotations = json.load(f)
        
        # Load class names
        classes_file = self.root_dir / 'classes.json'
        if classes_file.exists():
            with open(classes_file, 'r') as f:
                self.classes = json.load(f)
        else:
            # Infer classes from annotations
            self.classes = sorted(list(set(ann['class_name'] for ann in self.annotations)))
        
        self.num_classes = len(self.classes)
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}
        
        # Initialize loaders
        video_config = self.config.get('video', {})
        audio_config = self.config.get('audio', {})
        
        self.video_loader = VideoLoader(
            target_fps=video_config.get('fps', 30),
            target_size=tuple(video_config.get('size', [224, 224])),
            max_frames=video_config.get('max_frames', None),
            normalize=True
        )
        
        self.audio_extractor = AudioExtractor(
            sample_rate=audio_config.get('sample_rate', 16000),
            mono=audio_config.get('mono', True),
            normalize=True
        )
        
        # Augmentation
        if transform is None:
            self.transform = MultiModalAugmentation(mode=split)
        else:
            self.transform = transform
        
        # Processing config
        self.num_frames = video_config.get('num_frames', 16)
        self.audio_duration = audio_config.get('duration', 10.0)
        self.audio_length = int(self.audio_extractor.sample_rate * self.audio_duration)
        
        print(f"Loaded {split} dataset: {len(self)} samples, {self.num_classes} classes")
    
    def __len__(self) -> int:
        """Return dataset size."""
        return len(self.annotations)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a single sample.
        
        Returns:
            Dictionary containing:
                - video: Video frames tensor (T, C, H, W)
                - audio: Audio waveform tensor (1, samples)
                - label: Class label (int)
                - video_path: Path to video file
        """
        annotation = self.annotations[idx]
        video_path = annotation['video_path']
        label = annotation['label']
        
        # Make path absolute if relative
        if not Path(video_path).is_absolute():
            video_path = str(self.root_dir / video_path)
        
        try:
            # Load video
            if self.load_video:
                video_frames = self.video_loader.load_video(video_path)
                # Sample frames uniformly
                video_frames = uniform_temporal_sampling(video_frames, self.num_frames)
            else:
                # Create dummy video tensor
                video_frames = torch.zeros(self.num_frames, 3, 224, 224)
            
            # Load audio
            if self.load_audio:
                try:
                    audio_waveform = self.audio_extractor.extract_audio(video_path)
                    audio_waveform = pad_or_truncate(audio_waveform, self.audio_length)
                except Exception as e:
                    # Use silent audio if extraction fails
                    print(f"Warning: Audio extraction failed for {video_path}: {e}")
                    audio_waveform = torch.zeros(1, self.audio_length)
            else:
                # Create dummy audio tensor
                audio_waveform = torch.zeros(1, self.audio_length)
            
            # Apply augmentation
            video_frames, audio_waveform = self.transform(video_frames, audio_waveform)
            
            sample = {
                'video': video_frames,
                'audio': audio_waveform,
                'label': torch.tensor(label, dtype=torch.long),
                'video_path': video_path
            }
            
            # Add caption if available
            if 'caption' in annotation:
                sample['caption'] = annotation['caption']
            
            return sample
            
        except Exception as e:
            print(f"Error loading sample {idx} ({video_path}): {e}")
            # Return a dummy sample
            return {
                'video': torch.zeros(self.num_frames, 3, 224, 224),
                'audio': torch.zeros(1, self.audio_length),
                'label': torch.tensor(0, dtype=torch.long),
                'video_path': video_path
            }
    
    def get_class_name(self, label: int) -> str:
        """Get class name from label index."""
        return self.classes[label]


def create_dataloader(
    dataset: Dataset,
    batch_size: int = 16,
    shuffle: bool = True,
    num_workers: int = 4,
    pin_memory: bool = True,
    drop_last: bool = False
) -> DataLoader:
    """
    Create a DataLoader for the dataset.
    
    Args:
        dataset: PyTorch dataset
        batch_size: Batch size
        shuffle: Shuffle data
        num_workers: Number of worker processes
        pin_memory: Pin memory for faster GPU transfer
        drop_last: Drop last incomplete batch
        
    Returns:
        DataLoader instance
    """
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
        collate_fn=collate_multimodal
    )


def collate_multimodal(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """
    Collate function for multi-modal data.
    
    Args:
        batch: List of samples
        
    Returns:
        Batched dictionary
    """
    video = torch.stack([item['video'] for item in batch])
    audio = torch.stack([item['audio'] for item in batch])
    labels = torch.stack([item['label'] for item in batch])
    
    collated = {
        'video': video,
        'audio': audio,
        'labels': labels,
        'video_paths': [item['video_path'] for item in batch]
    }
    
    # Add captions if available
    if 'caption' in batch[0]:
        collated['captions'] = [item['caption'] for item in batch]
    
    return collated
