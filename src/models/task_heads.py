"""Task-specific heads for multi-task learning."""
import torch
import torch.nn as nn
from typing import List


class ClassificationHead(nn.Module):
    """Classification head for action recognition."""
    
    def __init__(
        self,
        input_dim: int = 512,
        hidden_dims: List[int] = [512, 256],
        num_classes: int = 400,
        dropout: float = 0.3
    ):
        """
        Initialize classification head.
        
        Args:
            input_dim: Input feature dimension
            hidden_dims: Hidden layer dimensions
            num_classes: Number of output classes
            dropout: Dropout rate
        """
        super().__init__()
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.BatchNorm1d(hidden_dim)
            ])
            prev_dim = hidden_dim
        
        # Output layer
        layers.append(nn.Linear(prev_dim, num_classes))
        
        self.classifier = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input features (B, D) or (B, T, D)
            
        Returns:
            Class logits (B, num_classes)
        """
        # If temporal features, take mean
        if x.dim() == 3:
            x = torch.mean(x, dim=1)
        
        return self.classifier(x)


class CaptioningHead(nn.Module):
    """Video captioning head using simple decoder."""
    
    def __init__(
        self,
        input_dim: int = 512,
        hidden_dim: int = 512,
        vocab_size: int = 50257,  # GPT-2 vocab size
        num_layers: int = 2,
        max_length: int = 50,
        dropout: float = 0.2
    ):
        """
        Initialize captioning head.
        
        Args:
            input_dim: Input feature dimension
            hidden_dim: Hidden dimension
            vocab_size: Vocabulary size
            num_layers: Number of decoder layers
            max_length: Maximum caption length
            dropout: Dropout rate
        """
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.max_length = max_length
        
        # Project video features to hidden dim
        self.feature_proj = nn.Linear(input_dim, hidden_dim)
        
        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        
        # LSTM decoder
        self.decoder = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )
        
        # Output projection
        self.output_proj = nn.Linear(hidden_dim, vocab_size)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self,
        video_features: torch.Tensor,
        captions: torch.Tensor = None,
        teacher_forcing_ratio: float = 1.0
    ) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            video_features: Video features (B, D) or (B, T, D)
            captions: Target captions (B, seq_len) for training
            teacher_forcing_ratio: Probability of using teacher forcing
            
        Returns:
            Caption logits (B, seq_len, vocab_size)
        """
        batch_size = video_features.size(0)
        
        # Pool temporal features if needed
        if video_features.dim() == 3:
            video_features = torch.mean(video_features, dim=1)
        
        # Project features
        video_features = self.feature_proj(video_features)
        
        # Initialize decoder state with video features
        h0 = video_features.unsqueeze(0).repeat(self.decoder.num_layers, 1, 1)
        c0 = torch.zeros_like(h0)
        
        if captions is not None:
            # Training mode with teacher forcing
            seq_len = captions.size(1)
            outputs = []
            
            # Start token (assume 0 is <BOS>)
            input_token = torch.zeros(batch_size, 1, dtype=torch.long, device=video_features.device)
            
            hidden = (h0, c0)
            
            for t in range(seq_len):
                # Embed input
                embedded = self.embedding(input_token).squeeze(1)
                embedded = self.dropout(embedded)
                
                # Decoder step
                output, hidden = self.decoder(embedded.unsqueeze(1), hidden)
                
                # Project to vocabulary
                logits = self.output_proj(output.squeeze(1))
                outputs.append(logits)
                
                # Teacher forcing
                if torch.rand(1).item() < teacher_forcing_ratio:
                    input_token = captions[:, t].unsqueeze(1)
                else:
                    input_token = logits.argmax(dim=-1).unsqueeze(1)
            
            return torch.stack(outputs, dim=1)
        
        else:
            # Inference mode - autoregressive generation
            outputs = []
            input_token = torch.zeros(batch_size, 1, dtype=torch.long, device=video_features.device)
            hidden = (h0, c0)
            
            for t in range(self.max_length):
                embedded = self.embedding(input_token).squeeze(1)
                output, hidden = self.decoder(embedded.unsqueeze(1), hidden)
                logits = self.output_proj(output.squeeze(1))
                
                outputs.append(logits)
                input_token = logits.argmax(dim=-1).unsqueeze(1)
            
            return torch.stack(outputs, dim=1)
    
    def generate(
        self,
        video_features: torch.Tensor,
        max_length: int = None,
        temperature: float = 1.0
    ) -> torch.Tensor:
        """
        Generate captions greedily.
        
        Args:
            video_features: Video features (B, D)
            max_length: Maximum generation length
            temperature: Sampling temperature
            
        Returns:
            Generated token IDs (B, seq_len)
        """
        if max_length is None:
            max_length = self.max_length
        
        self.eval()
        with torch.no_grad():
            logits = self.forward(video_features, captions=None)
            
            if temperature != 1.0:
                logits = logits / temperature
            
            tokens = logits.argmax(dim=-1)
        
        return tokens
