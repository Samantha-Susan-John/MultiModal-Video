"""Video loading and frame extraction utilities."""
import cv2
import numpy as np
from pathlib import Path
from typing import Optional, Tuple, List
import torch


class VideoLoader:
    """Load and preprocess video frames."""
    
    def __init__(
        self,
        target_fps: int = 30,
        target_size: Tuple[int, int] = (224, 224),
        max_frames: Optional[int] = None,
        normalize: bool = True
    ):
        """
        Initialize video loader.
        
        Args:
            target_fps: Target frames per second for sampling
            target_size: Target frame size (height, width)
            max_frames: Maximum number of frames to load
            normalize: Whether to normalize pixel values to [0, 1]
        """
        self.target_fps = target_fps
        self.target_size = target_size
        self.max_frames = max_frames
        self.normalize = normalize
    
    def load_video(
        self, 
        video_path: str, 
        start_time: float = 0.0,
        duration: Optional[float] = None
    ) -> torch.Tensor:
        """
        Load video and return frames as tensor.
        
        Args:
            video_path: Path to video file
            start_time: Start time in seconds
            duration: Duration to load in seconds
            
        Returns:
            Tensor of shape (T, C, H, W) where T is number of frames
        """
        if not Path(video_path).exists():
            raise FileNotFoundError(f"Video not found: {video_path}")
        
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")
        
        # Get video properties
        original_fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Calculate frame indices to sample
        frame_indices = self._get_frame_indices(
            original_fps, total_frames, start_time, duration
        )
        
        frames = []
        for idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            
            if not ret:
                break
            
            # Preprocess frame
            frame = self._preprocess_frame(frame)
            frames.append(frame)
        
        cap.release()
        
        if len(frames) == 0:
            raise ValueError(f"No frames loaded from video: {video_path}")
        
        # Convert to tensor: (T, H, W, C) -> (T, C, H, W)
        frames = np.stack(frames, axis=0)
        frames = torch.from_numpy(frames).permute(0, 3, 1, 2).float()
        
        if self.normalize:
            frames = frames / 255.0
        
        return frames
    
    def _get_frame_indices(
        self,
        original_fps: float,
        total_frames: int,
        start_time: float,
        duration: Optional[float]
    ) -> List[int]:
        """Calculate which frame indices to sample."""
        # Calculate start and end frames
        start_frame = int(start_time * original_fps)
        
        if duration is not None:
            end_frame = min(start_frame + int(duration * original_fps), total_frames)
        else:
            end_frame = total_frames
        
        # Sample frames at target FPS
        fps_ratio = original_fps / self.target_fps
        frame_indices = []
        
        current_frame = start_frame
        while current_frame < end_frame:
            frame_indices.append(int(current_frame))
            current_frame += fps_ratio
            
            if self.max_frames and len(frame_indices) >= self.max_frames:
                break
        
        return frame_indices
    
    def _preprocess_frame(self, frame: np.ndarray) -> np.ndarray:
        """Preprocess a single frame."""
        # Convert BGR to RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Resize
        frame = cv2.resize(frame, self.target_size, interpolation=cv2.INTER_LINEAR)
        
        return frame
    
    def get_video_info(self, video_path: str) -> dict:
        """Get video metadata."""
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")
        
        info = {
            'fps': cap.get(cv2.CAP_PROP_FPS),
            'frame_count': int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
            'width': int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            'height': int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            'duration': cap.get(cv2.CAP_PROP_FRAME_COUNT) / cap.get(cv2.CAP_PROP_FPS)
        }
        
        cap.release()
        return info


def uniform_temporal_sampling(
    frames: torch.Tensor,
    num_samples: int
) -> torch.Tensor:
    """
    Uniformly sample frames from a video.
    
    Args:
        frames: Tensor of shape (T, C, H, W)
        num_samples: Number of frames to sample
        
    Returns:
        Sampled frames of shape (num_samples, C, H, W)
    """
    T = frames.shape[0]
    
    if T <= num_samples:
        # Repeat frames if not enough
        indices = torch.linspace(0, T - 1, num_samples).long()
    else:
        # Sample uniformly
        indices = torch.linspace(0, T - 1, num_samples).long()
    
    return frames[indices]
