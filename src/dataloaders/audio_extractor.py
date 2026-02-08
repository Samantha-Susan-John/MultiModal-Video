"""Audio extraction and preprocessing from video files."""
import subprocess
import tempfile
from pathlib import Path
from typing import Optional, Tuple
import torch
import torchaudio
import numpy as np


class AudioExtractor:
    """Extract and preprocess audio from video files."""
    
    def __init__(
        self,
        sample_rate: int = 16000,
        mono: bool = True,
        normalize: bool = True
    ):
        """
        Initialize audio extractor.
        
        Args:
            sample_rate: Target sample rate
            mono: Convert to mono audio
            normalize: Normalize audio amplitude
        """
        self.sample_rate = sample_rate
        self.mono = mono
        self.normalize = normalize
    
    def extract_audio(
        self,
        video_path: str,
        start_time: float = 0.0,
        duration: Optional[float] = None
    ) -> torch.Tensor:
        """
        Extract audio from video file.
        
        Args:
            video_path: Path to video file
            start_time: Start time in seconds
            duration: Duration in seconds
            
        Returns:
            Audio waveform tensor of shape (channels, samples)
        """
        if not Path(video_path).exists():
            raise FileNotFoundError(f"Video not found: {video_path}")
        
        # Try to load audio directly with torchaudio
        try:
            waveform, sr = torchaudio.load(video_path)
        except Exception:
            # If direct loading fails, extract audio using ffmpeg
            waveform, sr = self._extract_with_ffmpeg(video_path)
        
        # Resample if needed
        if sr != self.sample_rate:
            resampler = torchaudio.transforms.Resample(sr, self.sample_rate)
            waveform = resampler(waveform)
        
        # Convert to mono if needed
        if self.mono and waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        
        # Extract segment
        if start_time > 0 or duration is not None:
            start_sample = int(start_time * self.sample_rate)
            
            if duration is not None:
                end_sample = start_sample + int(duration * self.sample_rate)
                waveform = waveform[:, start_sample:end_sample]
            else:
                waveform = waveform[:, start_sample:]
        
        # Normalize
        if self.normalize:
            waveform = self._normalize_audio(waveform)
        
        return waveform
    
    def _extract_with_ffmpeg(self, video_path: str) -> Tuple[torch.Tensor, int]:
        """Extract audio using ffmpeg command."""
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
            tmp_path = tmp_file.name
        
        try:
            # Use ffmpeg to extract audio
            cmd = [
                'ffmpeg',
                '-i', video_path,
                '-vn',  # No video
                '-acodec', 'pcm_s16le',  # PCM encoding
                '-ar', str(self.sample_rate),  # Sample rate
                '-ac', '1' if self.mono else '2',  # Channels
                '-y',  # Overwrite
                tmp_path
            ]
            
            subprocess.run(
                cmd,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=True
            )
            
            # Load extracted audio
            waveform, sr = torchaudio.load(tmp_path)
            
            return waveform, sr
            
        finally:
            # Clean up temporary file
            Path(tmp_path).unlink(missing_ok=True)
    
    def _normalize_audio(self, waveform: torch.Tensor) -> torch.Tensor:
        """Normalize audio waveform."""
        # Simple peak normalization
        max_val = torch.abs(waveform).max()
        if max_val > 0:
            waveform = waveform / max_val
        return waveform
    
    def get_audio_info(self, video_path: str) -> dict:
        """Get audio metadata."""
        try:
            waveform, sr = torchaudio.load(video_path)
            
            info = {
                'sample_rate': sr,
                'num_channels': waveform.shape[0],
                'num_samples': waveform.shape[1],
                'duration': waveform.shape[1] / sr
            }
            
            return info
        except Exception as e:
            return {'error': str(e)}


def pad_or_truncate(
    waveform: torch.Tensor,
    target_length: int,
    padding_value: float = 0.0
) -> torch.Tensor:
    """
    Pad or truncate audio to target length.
    
    Args:
        waveform: Audio tensor of shape (channels, samples)
        target_length: Target number of samples
        padding_value: Value to use for padding
        
    Returns:
        Audio tensor of shape (channels, target_length)
    """
    current_length = waveform.shape[1]
    
    if current_length < target_length:
        # Pad
        padding = target_length - current_length
        waveform = torch.nn.functional.pad(waveform, (0, padding), value=padding_value)
    elif current_length > target_length:
        # Truncate
        waveform = waveform[:, :target_length]
    
    return waveform
